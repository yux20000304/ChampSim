/*
 *    Copyright 2023 The ChampSim Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*! @file
 *  This is an example of the PIN tool that demonstrates some basic PIN APIs
 *  and could serve as the starting point for developing your first PIN tool
 */

#include <cassert>
#include <cstddef>
#if defined(__has_include)
#if __has_include(<features.h>)
#include <features.h>
#endif
#endif
#ifndef __GLIBC_PREREQ
#define __GLIBC_PREREQ(a, b) 0
#endif
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <sys/resource.h>

#include "../../inc/trace_instruction.h"
#include "pin.H"

using std::find;
using std::string;
using trace_instr_format_t = input_instr;
struct ThreadData {
  std::ofstream* outfile;
  trace_instr_format_t curr_instr;
  UINT64 instrCount;
  boost::unordered_set<ADDRINT> footprint_line_set;
  boost::unordered_set<ADDRINT> footprint_page_set;
  ADDRINT stack_start;
  ADDRINT stack_end;
  BOOL inROI;
  BOOL should_write;
};

/* ================================================================== */
// Global variables
/* ================================================================== */

UINT64 instrCount = 0;

std::ofstream outfile;
boost::unordered_map<ADDRINT, UINT64> global_footprint_line_set;
boost::unordered_map<ADDRINT, UINT64> global_footprint_page_set;

// trace_instr_format_t curr_instr;
// TLS_KEY tls_key;
const UINT64 MAX_THREADS = 128;
ThreadData* thread_data[MAX_THREADS];

/* ===================================================================== */
// Command line switches
/* ===================================================================== */
KNOB<std::string> KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool", "o", "champsim.trace", "specify file name for Champsim tracer output");

KNOB<UINT64> KnobSkipInstructions(KNOB_MODE_WRITEONCE, "pintool", "s", "0", "How many instructions to skip before tracing begins");

KNOB<UINT64> KnobTraceInstructions(KNOB_MODE_WRITEONCE, "pintool", "t", "1000000", "How many instructions to trace");

KNOB<UINT64> KnobNumThreads(KNOB_MODE_WRITEONCE, "pintool", "n", "8", "Number of threads");

// const UINT64 MAX_THREADS = 128;
// BOOL inROI[MAX_THREADS];

/* ===================================================================== */
// Utilities
/* ===================================================================== */

/*!
 *  Print out help message.
 */
INT32 Usage()
{
  std::cerr << "This tool creates a register and memory access trace" << std::endl
            << "Specify the output trace file with -o" << std::endl
            << "Specify the number of instructions to skip before tracing with -s" << std::endl
            << "Specify the number of instructions to trace with -t" << std::endl
            << std::endl;

  std::cerr << KNOB_BASE::StringKnobSummary() << std::endl;

  return -1;
}

VOID ThreadStart(THREADID threadid, CONTEXT* ctxt, INT32 flags, VOID* v)
{
  thread_data[threadid] = new ThreadData;
  thread_data[threadid]->inROI = FALSE;
  assert(thread_data[threadid] != NULL);

  // 构造文件名，如 "champsim.trace.3" 表示线程 3 的 dump 文件
  std::stringstream ss;
  ss << KnobOutputFile.Value() << "." << threadid;
  thread_data[threadid]->outfile = new std::ofstream(ss.str().c_str(), std::ios::binary | std::ios::trunc);
  if (!thread_data[threadid]->outfile->is_open()) {
    std::cerr << "Error: cannot open dump file " << ss.str() << " for thread " << threadid << std::endl;
    exit(1);
  }

  // Get stack bounds using actual system stack size
  ADDRINT stack_start = 0;
  ADDRINT stack_end = 0;

  if (ctxt != NULL) {
    // Try to get stack information from the context
    ADDRINT sp = PIN_GetContextReg(ctxt, REG_STACK_PTR);
    if (sp != 0) {
      // Get actual stack size from system
      struct rlimit rlim;
      if (getrlimit(RLIMIT_STACK, &rlim) == 0) {
        ADDRINT actual_stack_size = rlim.rlim_cur;
        std::cout << "Thread " << threadid << " actual stack size: " << actual_stack_size << " bytes" << std::endl;
        stack_start = sp - actual_stack_size;
        stack_end = sp;
      } else {
        // Fallback to default if getrlimit fails
        const ADDRINT default_stack_size = 8 * 1024 * 1024; // 8MB
        std::cout << "Thread " << threadid << " using default stack size: " << default_stack_size << " bytes" << std::endl;
        stack_start = sp - default_stack_size;
        stack_end = sp;
      }
    }
  }

  thread_data[threadid]->stack_start = stack_start;
  thread_data[threadid]->stack_end = stack_end;
  thread_data[threadid]->instrCount = 0;
  thread_data[threadid]->should_write = FALSE;

  std::cout << "Thread " << threadid << " started. Dump file: " << ss.str() << " stack start: " << (void*)(thread_data[threadid]->stack_start)
            << " stack end: " << (void*)(thread_data[threadid]->stack_end) << std::endl;
  // PIN_SetThreadData(tls_key, thread_data[threadid], PIN_ThreadId());
}

VOID merge_footprint_set(THREADID tid)
{
  ThreadData* tdata = thread_data[tid];
  if (tdata) {
    for (auto line : tdata->footprint_line_set) {
      if (global_footprint_line_set.find(line) == global_footprint_line_set.end()) {
        global_footprint_line_set.insert(std::make_pair(line, 1));
      } else {
        global_footprint_line_set[line]++;
      }
    }
    for (auto page : tdata->footprint_page_set) {
      if (global_footprint_page_set.find(page) == global_footprint_page_set.end()) {
        global_footprint_page_set.insert(std::make_pair(page, 1));
      } else {
        global_footprint_page_set[page]++;
      }
    }
  }
}

VOID ThreadFini(THREADID threadid, const CONTEXT* ctxt, INT32 code, VOID* v)
{
  ThreadData* tdata = thread_data[threadid];
  if (tdata) {
    if (tdata->outfile) {
      tdata->outfile->flush();
      tdata->outfile->close();
      delete tdata->outfile;
      tdata->outfile = NULL;
    }
    merge_footprint_set(threadid);
    std::cout << "Thread " << threadid << " finished. Total instructions processed: " << tdata->instrCount
              << " footprint line size: " << tdata->footprint_line_set.size() << " footprint page size: " << tdata->footprint_page_set.size() << std::endl;
    std::cout.flush();
    delete tdata;
    thread_data[threadid] = NULL;
  }
}

/* ===================================================================== */
// Analysis routines
/* ===================================================================== */

VOID BeginInstruction(VOID* ip, THREADID tid)
{
  ThreadData* tdata = thread_data[tid];
  if (!(tdata != NULL && tid < MAX_THREADS && tid >= 0)) {
    std::cout << "pin_magic_inst: tdata is NULL or tid is out of range" << std::endl;
    std::cout.flush();
    exit(1);
    return;
  }
  tdata->should_write = FALSE;
  if (tdata->inROI) {
    tdata->instrCount++;
    BOOL in_window = (tdata->instrCount > KnobSkipInstructions.Value()) &&
                     (tdata->instrCount <= (KnobTraceInstructions.Value() + KnobSkipInstructions.Value()));
    if (in_window) {
      tdata->should_write = TRUE;
      tdata->curr_instr = {};
      tdata->curr_instr.ip = reinterpret_cast<unsigned long long>(ip);
    }
  }
}

/* ===================================================================== */
// Instrumentation callbacks
/* ===================================================================== */

static VOID pin_magic_inst(THREADID tid, ADDRINT value, ADDRINT field)
{
  ThreadData* tdata = thread_data[tid];
  if (!(tdata != NULL && tid < MAX_THREADS && tid >= 0)) {
    std::cout << "pin_magic_inst: tdata is NULL or tid is out of range" << std::endl;
    std::cout.flush();
    exit(1);
    return;
  }
  switch (field) {
  case 0xEE: // ROI START
    tdata->inROI = true;
    asm volatile("" : : : "memory");
    std::cout << "ROI START (tid " << tid << ")" << std::endl;
    std::cout.flush();
    break;
  case 0xFF: // ROI END
    tdata->inROI = false;
    asm volatile("" : : : "memory");
    std::cout << "ROI END (tid " << tid << ")" << std::endl;
    std::cout.flush();
    break;
  default:
    break;
  }
  return;
}

void ProcessInstructionOperands(VOID* ip, THREADID tid, BOOL taken, BOOL is_branch)
{
  ThreadData* tdata = thread_data[tid];
  if (!(tdata != NULL && tid < MAX_THREADS && tid >= 0)) {
    std::cout << "pin_magic_inst: tdata is NULL or tid is out of range" << std::endl;
    std::cout.flush();
    exit(1);
    return;
  }
  if (!tdata->should_write)
    return;

  // branch
  if (is_branch) {
    tdata->curr_instr.is_branch = 1;
    tdata->curr_instr.branch_taken = taken;
  }
  // // regR
  // UINT32* regReadsPtr = reinterpret_cast<UINT32*>(regReads);
  // for (UINT32 i = 0; i < readRegCount; i++) {
  //   UINT32 regNum = regReadsPtr[i];
  //   unsigned char* begin = tdata->curr_instr.source_registers;
  //   unsigned char* end = tdata->curr_instr.source_registers + NUM_INSTR_SOURCES;
  //   auto set_end = std::find(begin, end, 0);
  //   auto found = std::find(begin, set_end, static_cast<unsigned char>(regNum));
  //   if (found == set_end && set_end != end) {        // Not found and space available
  //     *set_end = static_cast<unsigned char>(regNum); // Add to first available slot
  //   }
  // }

  // // regW
  // UINT32* regWritesPtr = reinterpret_cast<UINT32*>(regWrites);
  // for (UINT32 i = 0; i < writeRegCount; i++) {
  //   UINT32 regNum = regWritesPtr[i];
  //   unsigned char* begin = tdata->curr_instr.destination_registers;
  //   unsigned char* end = tdata->curr_instr.destination_registers + NUM_INSTR_DESTINATIONS;
  //   auto set_end = std::find(begin, end, 0);
  //   auto found = std::find(begin, set_end, static_cast<unsigned char>(regNum));
  //   if (found == set_end && set_end != end) {        // Not found and space available
  //     *set_end = static_cast<unsigned char>(regNum); // Add to first available slot
  //   }
  // }

  // write
  char buf[sizeof(trace_instr_format_t)];
  std::memcpy(buf, &tdata->curr_instr, sizeof(trace_instr_format_t));
  // std::cout << "Thread " << tid << " writing instruction to file" << buf[1] << std::endl;
  // std::cout.flush();
  if (!tdata->outfile) {
    std::cout << "ProcessInstructionOperands: tdata->outfile is NULL for tid " << tid << ". This should not happen inside an ROI." << std::endl;
    std::cout.flush();
    exit(1);
    return;
  }
  tdata->outfile->write(buf, sizeof(trace_instr_format_t));
  tdata->should_write = FALSE;
}

VOID WriteToSourceMemory(THREADID tid, UINT32 memOp, ADDRINT memEA)
{
  ThreadData* tdata = thread_data[tid];
  if (tdata && tdata->should_write) {
    unsigned long long int* begin = tdata->curr_instr.source_memory;
    unsigned long long int* end = tdata->curr_instr.source_memory + NUM_INSTR_SOURCES;
    auto set_end = std::find(begin, end, 0);
    auto found = std::find(begin, set_end, memEA);
    if (found == set_end && set_end != end) { // Not found and space available
      *set_end = memEA;                       // Add to first available slot
      ADDRINT cache_line_id = memEA / 64;
      tdata->footprint_line_set.insert(cache_line_id);
      ADDRINT page_id = memEA / 4096;
      tdata->footprint_page_set.insert(page_id);
    }
  }
}

VOID WriteToDestinationMemory(THREADID tid, UINT32 memOp, ADDRINT memEA)
{
  ThreadData* tdata = thread_data[tid];
  if (tdata && tdata->should_write) {
    unsigned long long int* begin = tdata->curr_instr.destination_memory;
    unsigned long long int* end = tdata->curr_instr.destination_memory + NUM_INSTR_DESTINATIONS;
    auto set_end = std::find(begin, end, 0);
    auto found = std::find(begin, set_end, memEA);
    if (found == set_end && set_end != end) { // Not found and space available
      *set_end = memEA;                       // Add to first available slot
      ADDRINT cache_line_id = memEA / 64;
      tdata->footprint_line_set.insert(cache_line_id);
      ADDRINT page_id = memEA / 4096;
      tdata->footprint_page_set.insert(page_id);
    }
  }
}

VOID WriteToSourceRegister(THREADID tid, UINT32 regNum)
{
  ThreadData* tdata = thread_data[tid];
  if (tdata && tdata->should_write) {
    unsigned char* begin = tdata->curr_instr.source_registers;
    unsigned char* end = tdata->curr_instr.source_registers + NUM_INSTR_SOURCES;
    auto set_end = std::find(begin, end, 0);
    auto found = std::find(begin, set_end, static_cast<unsigned char>(regNum));
    if (found == set_end && set_end != end) {        // Not found and space available
      *set_end = static_cast<unsigned char>(regNum); // Add to first available slot
    }
  }
}

VOID WriteToDestinationRegister(THREADID tid, UINT32 regNum)
{
  ThreadData* tdata = thread_data[tid];
  if (tdata && tdata->should_write) {
    unsigned char* begin = tdata->curr_instr.destination_registers;
    unsigned char* end = tdata->curr_instr.destination_registers + NUM_INSTR_DESTINATIONS;
    auto set_end = std::find(begin, end, 0);
    auto found = std::find(begin, set_end, static_cast<unsigned char>(regNum));
    if (found == set_end && set_end != end) {        // Not found and space available
      *set_end = static_cast<unsigned char>(regNum); // Add to first available slot
    }
  }
}

// Is called for every instruction and instruments reads and writes
VOID Instruction(INS ins, VOID* v)
{
  if (INS_IsXchg(ins) && INS_OperandReg(ins, 0) == REG_RBX && INS_OperandReg(ins, 1) == REG_RBX) {
    std::cout << "xchg rbx rbx found" << std::endl;
    std::cout.flush();
    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)pin_magic_inst, IARG_THREAD_ID, IARG_REG_VALUE, REG_RBX, IARG_REG_VALUE, REG_RCX, IARG_END);
  }

  INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)BeginInstruction, IARG_INST_PTR, IARG_THREAD_ID, IARG_END);

  UINT32 readRegCount = INS_MaxNumRRegs(ins);
  for (UINT32 i = 0; i < readRegCount; i++) {
    UINT32 regNum = INS_RegR(ins, i);
    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)WriteToSourceRegister, IARG_THREAD_ID, IARG_UINT32, regNum, IARG_END);
  }

  UINT32 writeRegCount = INS_MaxNumWRegs(ins);
  for (UINT32 i = 0; i < writeRegCount; i++) {
    UINT32 regNum = INS_RegW(ins, i);
    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)WriteToDestinationRegister, IARG_THREAD_ID, IARG_UINT32, regNum, IARG_END);
  }

  UINT32 memOperands = INS_MemoryOperandCount(ins);
  for (UINT32 memOp = 0; memOp < memOperands; memOp++) {
    // std::cout << "Thread " << IARG_THREAD_ID << ":" << PIN_GetTid() << " of instruction at " << INS_Address(ins) << " is "
    //           << (INS_MemoryOperandIsRead(ins, memOp) ? "read" : "write") << std::endl;
    if (INS_MemoryOperandIsRead(ins, memOp)) {
      INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)WriteToSourceMemory, IARG_THREAD_ID, IARG_UINT32, memOp, IARG_MEMORYOP_EA, memOp, IARG_END);
    }
    if (INS_MemoryOperandIsWritten(ins, memOp)) {
      INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)WriteToDestinationMemory, IARG_THREAD_ID, IARG_UINT32, memOp, IARG_MEMORYOP_EA, memOp, IARG_END);
    }
  }

  BOOL is_branch = INS_IsBranch(ins);

  INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)ProcessInstructionOperands, IARG_INST_PTR, IARG_THREAD_ID, IARG_BRANCH_TAKEN, IARG_BOOL, is_branch, IARG_END);
}

/*!
 * Print out analysis results.
 * This function is called when the application exits.
 * @param[in]   code            exit code of the application
 * @param[in]   v               value specified by the tool in the
 *                              PIN_AddFiniFunction function call
 */
VOID Fini(INT32 code, VOID* v)
{
  UINT64 shared_line_count = 0;
  UINT64 shared_page_count = 0;
  for (auto line : global_footprint_line_set) {
    if (line.second > 1) {
      shared_line_count++;
    }
  }
  for (auto page : global_footprint_page_set) {
    if (page.second > 1) {
      shared_page_count++;
    }
  }
  std::cout << "Shared line count: " << shared_line_count * 64ul / 1024ul / 1024ul << " MB" << std::endl;
  std::cout << "Shared page count: " << shared_page_count * 4096ul / 1024ul / 1024ul << " MB" << std::endl;
  outfile.close();
}

/*!
 * The main procedure of the tool.
 * This function is called when the application image is loaded but not yet started.
 * @param[in]   argc            total number of elements in the argv array
 * @param[in]   argv            array of command line arguments,
 *                              including pin -t <toolname> -- ...
 */
int main(int argc, char* argv[])
{
  // Initialize PIN library. Print help message if -h(elp) is specified
  // in the command line or the command line is invalid
  if (PIN_Init(argc, argv))
    return Usage();

  outfile.open(KnobOutputFile.Value().c_str(), std::ios_base::binary | std::ios_base::trunc);
  if (!outfile) {
    std::cout << "Couldn't open output trace file. Exiting." << std::endl;
    exit(1);
  }

  // tls_key = PIN_CreateThreadDataKey(NULL);
  // assert(tls_key != -1);
  // Register function to be called to instrument instructions
  INS_AddInstrumentFunction(Instruction, 0);

  PIN_AddThreadStartFunction(ThreadStart, 0);
  PIN_AddThreadFiniFunction((THREAD_FINI_CALLBACK)ThreadFini, 0);

  // Register function to be called when the application exits
  PIN_AddFiniFunction(Fini, 0);
  // Start the program, never returns
  PIN_StartProgram();

  return 0;
}
