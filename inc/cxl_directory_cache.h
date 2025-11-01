#pragma once

#include <deque>
#include <list>
#include <unordered_map>
#include <vector>

#include "chrono.h"

struct cxl_directory_cache {
  struct node {
    std::size_t page{};
    std::vector<std::uint64_t> words{};
  };

  std::size_t capacity{0};
  std::size_t words_per_entry{0};
  champsim::chrono::clock::duration read_penalty{champsim::chrono::clock::duration::zero()};
  std::size_t max_requests_per_cycle{0};
  std::list<node> lru{};
  std::unordered_map<std::size_t, std::list<node>::iterator> map{};
  std::deque<champsim::chrono::clock::time_point> busy_until{};

  void clear()
  {
    lru.clear();
    map.clear();
  }

  [[nodiscard]] bool enabled() const { return capacity > 0 && words_per_entry > 0; }

  std::vector<std::uint64_t>* find_entry(std::size_t page_index);
  std::vector<std::uint64_t>* insert_entry(std::size_t page_index, std::vector<std::uint64_t> words);
  std::vector<std::uint64_t> evict_entry();
  champsim::chrono::clock::duration enforce_bandwidth(champsim::chrono::clock::time_point now,
                                                      champsim::chrono::clock::duration service_time);
};
