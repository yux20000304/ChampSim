#include "cxl_directory_cache.h"

#include <algorithm>

std::vector<std::uint64_t>* cxl_directory_cache::find_entry(std::size_t page_index)
{
  auto map_it = map.find(page_index);
  if (map_it == std::end(map)) {
    return nullptr;
  }

  lru.splice(std::begin(lru), lru, map_it->second);
  return &map_it->second->words;
}

std::vector<std::uint64_t>* cxl_directory_cache::insert_entry(std::size_t page_index, std::vector<std::uint64_t> words)
{
  auto map_it = map.find(page_index);
  if (map_it != std::end(map)) {
    map_it->second->words = std::move(words);
    lru.splice(std::begin(lru), lru, map_it->second);
    return &map_it->second->words;
  }

  if (map.size() >= capacity && !lru.empty()) {
    auto victim_it = std::prev(lru.end());
    map.erase(victim_it->page);
    lru.pop_back();
  }

  lru.emplace_front(node{page_index, std::move(words)});
  map[page_index] = lru.begin();
  return &lru.begin()->words;
}

std::vector<std::uint64_t> cxl_directory_cache::evict_entry()
{
  if (lru.empty()) {
    return {};
  }

  auto victim_it = std::prev(lru.end());
  std::vector<std::uint64_t> words = std::move(victim_it->words);
  map.erase(victim_it->page);
  lru.pop_back();
  return words;
}

champsim::chrono::clock::duration cxl_directory_cache::enforce_bandwidth(champsim::chrono::clock::time_point now,
                                                                         champsim::chrono::clock::duration service_time)
{
  if (max_requests_per_cycle == 0) {
    return champsim::chrono::clock::duration::zero();
  }

  while (!busy_until.empty() && busy_until.front() <= now) {
    busy_until.pop_front();
  }

  champsim::chrono::clock::duration wait{};
  if (busy_until.size() >= max_requests_per_cycle) {
    auto earliest = busy_until.front();
    wait = earliest - now;
    now = earliest;
    while (!busy_until.empty() && busy_until.front() <= now) {
      busy_until.pop_front();
    }
  }

  busy_until.push_back(now + service_time);
  return wait;
}
