#ifndef __SIMPLE_CAM_EVENT_LOOP_H__
#define __SIMPLE_CAM_EVENT_LOOP_H__

#include <atomic>
#include <deque>
#include <mutex>
#include <type_traits>
#include <new>
#include <utility>
#include <cstddef>
#include <cstring>

// For evutil_socket_t in the wakeup callback signature.
#include <event2/util.h>

struct event_base;
struct event;

class EventLoop
{
public:
	EventLoop();
	~EventLoop();

	void exit(int code = 0);
	int exec();

	template <typename F>
	void callLater(F &&func)
	{
		enqueue(Task(std::forward<F>(func)));
	}

private:
	EventLoop *instance_ = nullptr;

	// Move-only, small-buffer callable to avoid heap allocations for typical lambdas.
	template <std::size_t InlineBytes>
	class InplaceFunction
	{
	public:
		InplaceFunction() = default;
		InplaceFunction(std::nullptr_t) {}

		InplaceFunction(const InplaceFunction &) = delete;
		InplaceFunction &operator=(const InplaceFunction &) = delete;

		InplaceFunction(InplaceFunction &&other) noexcept { moveFrom(std::move(other)); }
		InplaceFunction &operator=(InplaceFunction &&other) noexcept
		{
			if (this != &other)
			{
				reset();
				moveFrom(std::move(other));
			}
			return *this;
		}

		~InplaceFunction() { reset(); }

		template <typename F, typename = std::enable_if_t<std::is_invocable_r_v<void, F &>>>
		explicit InplaceFunction(F &&f)
		{
			using Fn = std::decay_t<F>;
			static_assert(std::is_move_constructible_v<Fn>, "callLater() functor must be move-constructible");

			if constexpr (sizeof(Fn) <= InlineBytes && alignof(Fn) <= alignof(Storage))
			{
				new (&storage_) Fn(std::forward<F>(f));
				invoke_ = [](void *p)
				{ (*reinterpret_cast<Fn *>(p))(); };
				destroy_ = [](void *p)
				{ reinterpret_cast<Fn *>(p)->~Fn(); };
				move_ = [](void *dst, void *src)
				{
					new (dst) Fn(std::move(*reinterpret_cast<Fn *>(src)));
					reinterpret_cast<Fn *>(src)->~Fn();
				};
			}
			else
			{
				Fn *heap = new Fn(std::forward<F>(f));
				std::memcpy(&storage_, &heap, sizeof(heap));
				invoke_ = [](void *p)
				{
					Fn *ptr;
					std::memcpy(&ptr, p, sizeof(ptr));
					(*ptr)();
				};
				destroy_ = [](void *p)
				{
					Fn *ptr;
					std::memcpy(&ptr, p, sizeof(ptr));
					delete ptr;
				};
				move_ = [](void *dst, void *src)
				{
					std::memcpy(dst, src, sizeof(Fn *));
					Fn *nullPtr = nullptr;
					std::memcpy(src, &nullPtr, sizeof(nullPtr));
				};
			}
		}

		explicit operator bool() const noexcept { return invoke_ != nullptr; }

		void operator()()
		{
			if (invoke_)
				invoke_(&storage_);
		}

		void reset()
		{
			if (destroy_)
				destroy_(&storage_);
			invoke_ = nullptr;
			destroy_ = nullptr;
			move_ = nullptr;
		}

	private:
		using Storage = std::aligned_storage_t<InlineBytes, alignof(std::max_align_t)>;
		Storage storage_;
		void (*invoke_)(void *) = nullptr;
		void (*destroy_)(void *) = nullptr;
		void (*move_)(void *dst, void *src) = nullptr;

		void moveFrom(InplaceFunction &&other) noexcept
		{
			invoke_ = other.invoke_;
			destroy_ = other.destroy_;
			move_ = other.move_;
			if (move_)
				move_(&storage_, &other.storage_);
			other.invoke_ = nullptr;
			other.destroy_ = nullptr;
			other.move_ = nullptr;
		}
	};

	using Task = InplaceFunction<128>;

	struct event_base *event_;
	struct event *wakeupEvent_;
	int wakeupFd_;
	std::atomic<bool> exit_;
	int exitCode_;

	std::deque<Task> calls_;
	std::mutex lock_;
	std::atomic<bool> wakeupPending_{false};

	void enqueue(Task &&task);

	void interrupt();
	void dispatchCalls();
	static void wakeupCb(evutil_socket_t, short, void *arg);
};

#endif /* __SIMPLE_CAM_EVENT_LOOP_H__ */
