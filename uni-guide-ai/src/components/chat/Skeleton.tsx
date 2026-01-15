import { motion } from 'framer-motion';

export function MessageSkeleton() {
  return (
    <div className="flex gap-3 animate-pulse">
      {/* Avatar skeleton */}
      <div className="flex-shrink-0 w-10 h-10 rounded-full skeleton" />
      
      {/* Message bubble skeleton */}
      <div className="flex-1 max-w-[65%] space-y-3">
        <div className="h-4 skeleton rounded-full w-3/4" />
        <div className="h-4 skeleton rounded-full w-full" />
        <div className="h-4 skeleton rounded-full w-2/3" />
      </div>
    </div>
  );
}

export function ConversationSkeleton() {
  return (
    <div className="flex items-center gap-3 px-3 py-2.5 rounded-xl animate-pulse">
      <div className="w-8 h-8 rounded-lg skeleton" />
      <div className="flex-1 space-y-2">
        <div className="h-4 skeleton rounded-full w-3/4" />
        <div className="h-3 skeleton rounded-full w-1/2" />
      </div>
    </div>
  );
}

export function SuggestedQuestionsSkeleton() {
  return (
    <div className="flex gap-3 overflow-hidden px-4">
      {[1, 2, 3, 4].map((i) => (
        <div
          key={i}
          className="flex-shrink-0 h-10 w-48 skeleton rounded-full"
        />
      ))}
    </div>
  );
}

export function WelcomeSkeleton() {
  return (
    <div className="text-center py-12 space-y-6">
      <div className="w-20 h-20 mx-auto rounded-full skeleton" />
      <div className="space-y-3 max-w-md mx-auto">
        <div className="h-8 skeleton rounded-full w-3/4 mx-auto" />
        <div className="h-4 skeleton rounded-full w-full" />
        <div className="h-4 skeleton rounded-full w-2/3 mx-auto" />
      </div>
    </div>
  );
}