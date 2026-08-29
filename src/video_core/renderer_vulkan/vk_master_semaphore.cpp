// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <limits>
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"

#include "common/assert.h"

namespace Vulkan {

constexpr u64 WAIT_TIMEOUT = std::numeric_limits<u64>::max();

MasterSemaphore::MasterSemaphore(const Instance& instance_) : instance{instance_} {
    const vk::StructureChain semaphore_chain = {
        vk::SemaphoreCreateInfo{},
        vk::SemaphoreTypeCreateInfo{
            .semaphoreType = vk::SemaphoreType::eTimeline,
            .initialValue = 0,
        },
    };
    auto [semaphore_result, sem] =
        instance.GetDevice().createSemaphoreUnique(semaphore_chain.get());
    ASSERT_MSG(semaphore_result == vk::Result::eSuccess, "Failed to create master semaphore: {}",
               vk::to_string(semaphore_result));
    semaphore = std::move(sem);
}

MasterSemaphore::~MasterSemaphore() = default;

void MasterSemaphore::Refresh() {
    u64 this_tick{};
    u64 counter{};
    do {
        this_tick = gpu_tick.load(std::memory_order_acquire);
        auto [counter_result, cntr] = instance.GetDevice().getSemaphoreCounterValue(*semaphore);
        ASSERT_MSG(counter_result == vk::Result::eSuccess,
                   "Failed to get master semaphore value: {}", vk::to_string(counter_result));
        counter = cntr;
        if (counter < this_tick) {
            return;
        }
    } while (!gpu_tick.compare_exchange_weak(this_tick, counter, std::memory_order_release,
                                             std::memory_order_relaxed));
}

void MasterSemaphore::Wait(u64 tick) {
    // No need to wait if the GPU is ahead of the tick
    if (IsFree(tick)) {
        return;
    }
    // Update the GPU tick and try again
    Refresh();
    if (IsFree(tick)) {
        return;
    }

    // If none of the above is hit, fallback to a regular wait
    const vk::SemaphoreWaitInfo wait_info = {
        .semaphoreCount = 1,
        .pSemaphores = &semaphore.get(),
        .pValues = &tick,
    };

    // diag-v3: bounded poll so a GPU that never signals produces a log instead of hanging forever
    constexpr u64 poll_timeout = 50000000ull; // 50 ms
    auto wait_start = std::chrono::steady_clock::now();
    auto last_log = wait_start;
    while (true) {
        const auto wait_result = instance.GetDevice().waitSemaphores(&wait_info, poll_timeout);
        if (wait_result == vk::Result::eSuccess) {
            break;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - last_log > std::chrono::milliseconds(500)) {
            last_log = now;
            const u64 elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - wait_start).count();
            const auto [counter_result, counter] = instance.GetDevice().getSemaphoreCounterValue(*semaphore);
            LOG_ERROR(Render_Vulkan,
                      "[WAITSEM] waiting tick={} current_counter={} elapsed={}ms counter_result={}", tick, counter, elapsed, vk::to_string(counter_result));
        }
    }
    Refresh();

}

} // namespace Vulkan
