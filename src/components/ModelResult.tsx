import { LevelBadge } from "./LevelBadge";
import { Card } from "@/components/ui/card";

interface ModelResultProps {
  modelName: string;
  level: 1 | 2 | 3 | 4 | 5 | 6;
  confidence?: number;
}

export const ModelResult = ({ modelName, level, confidence }: ModelResultProps) => {
  return (
    <Card className="p-4 bg-card border-border hover:border-primary/50 transition-all duration-300 hover:shadow-lg">
      <div className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-foreground">{modelName}</h3>
          {confidence && (
            <span className="text-sm text-muted-foreground">
              {(confidence * 100).toFixed(1)}%
            </span>
          )}
        </div>
        <LevelBadge level={level} />
      </div>
    </Card>
  );
};
