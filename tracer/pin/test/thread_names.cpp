#include <iostream>
#include <string>
#include <thread>
#include <vector>

void print_name(const std::string& name)
{
  uint64_t dummyval = 0;
  __asm__ __volatile__("xchg %%rbx, %%rbx;" ::"b"((uint64_t)(&dummyval)), "c"((uint64_t)0xEE));
  for(int i = 0 ; i < 100000 ; i++)
    std::cout << i << " Hello from thread " << name << std::endl;
  __asm__ __volatile__("xchg %%rbx, %%rbx;" ::"b"((uint64_t)(&dummyval)), "c"((uint64_t)0xFF));
}

int main()
{
  std::vector<std::string> names{"Alice", "Bob", "Charlie", "Diana"};
  std::vector<std::thread> threads;
  threads.reserve(names.size());

  for (const auto& name : names) {
    threads.emplace_back(print_name, name);
  }

  for (auto& t : threads) {
    t.join();
  }

  return 0;
}
