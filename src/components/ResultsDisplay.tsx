import { ModelResult } from "./ModelResult";
import { LevelBadge } from "./LevelBadge";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface ModelPrediction {
  modelName: string;
  level: 1 | 2 | 3 | 4 | 5 | 6;
  confidence: number;
}

interface ResultsDisplayProps {
  word: string;
  predictions: ModelPrediction[];
  hardVote: 1 | 2 | 3 | 4 | 5 | 6;
}

export const ResultsDisplay = ({ word, predictions, hardVote }: ResultsDisplayProps) => {
  return (
    <div className="w-full max-w-6xl mx-auto animate-in fade-in duration-500">
      <div className="mb-8 text-center">
        <h2 className="text-3xl font-bold text-foreground mb-2">
          نتائج التصنيف لـ: <span className="text-primary">{word}</span>
        </h2>
      </div>

      <Tabs defaultValue="all" className="w-full">
        <TabsList className="grid w-full max-w-md mx-auto grid-cols-2 mb-8">
          <TabsTrigger value="all">جميع النماذج</TabsTrigger>
          <TabsTrigger value="consensus">النتيجة النهائية</TabsTrigger>
        </TabsList>

        <TabsContent value="all" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {predictions.map((prediction, index) => (
              <ModelResult
                key={index}
                modelName={prediction.modelName}
                level={prediction.level}
                confidence={prediction.confidence}
              />
            ))}
          </div>
        </TabsContent>

        <TabsContent value="consensus">
          <Card className="p-8 bg-card border-2 border-primary/30 shadow-xl max-w-2xl mx-auto">
            <div className="text-center space-y-4">
              <h3 className="text-2xl font-bold text-foreground">
                نتيجة التصويت الصعب
              </h3>
              <p className="text-muted-foreground mb-6">
                التصنيف الأكثر اتفاقًا بين جميع النماذج
              </p>
              <div className="flex justify-center">
                <LevelBadge level={hardVote} className="text-lg px-6 py-3" />
              </div>
            </div>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};
