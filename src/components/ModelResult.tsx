import { LevelBadge } from "./LevelBadge";
import { Card } from "@/components/ui/card";
interface ModelResultProps {
  modelName: string;
  level: 1 | 2 | 3 | 4 | 5 | 6;
  confidence?: number;
}
export const ModelResult = ({
  modelName,
  level,
  confidence
}: ModelResultProps) => {
  return <Card className="p-4 bg-card border-border hover:border-primary/50 transition-all duration-300 hover:shadow-lg">
      <div className="flex flex-col gap-3">
        <h3 className="font-semibold text-foreground text-center">{modelName}</h3>
        <LevelBadge level={level} className="text-center" />
      </div>
    </Card>;
};