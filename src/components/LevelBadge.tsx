import { cn } from "@/lib/utils";

interface LevelBadgeProps {
  level: 1 | 2 | 3 | 4 | 5 | 6;
  className?: string;
}

const levelLabels = {
  1: "مرحلة ما قبل المدرسة والصف 1-2",
  2: "الصف 3-4",
  3: "الصف 5-6",
  4: "الصف 7-9",
  5: "الصف 10-12",
  6: "الجامعة",
};

const levelColors = {
  1: "bg-level-1 text-level-1-foreground",
  2: "bg-level-2 text-level-2-foreground",
  3: "bg-level-3 text-level-3-foreground",
  4: "bg-level-4 text-level-4-foreground",
  5: "bg-level-5 text-level-5-foreground",
  6: "bg-level-6 text-level-6-foreground",
};

export const LevelBadge = ({ level, className }: LevelBadgeProps) => {
  return (
    <span
      className={cn(
        "inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold transition-all duration-300 shadow-md",
        levelColors[level],
        className
      )}
    >
      المستوى {level}: {levelLabels[level]}
    </span>
  );
};
