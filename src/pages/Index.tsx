import { useState } from "react";
import { ClassificationInput } from "@/components/ClassificationInput";
import { ResultsDisplay } from "@/components/ResultsDisplay";
import { useToast } from "@/hooks/use-toast";

// Mapping levels to colors
const levelColors: { [key: number]: string } = {
  1: "#1B4D3E",
  2: "#206040",
  3: "#258060",
  4: "#2B9A80",
  5: "#32B3A0",
  6: "#38CCB8",
};

// Function to call backend for Random Forest and random for other models
const classifyWord = async (word: string) => {
  try {
    // Fetch real Random Forest prediction
    const response = await fetch("http://localhost:8000/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ word }),
    });

    if (!response.ok) throw new Error("فشل الاتصال بالخادم");

    const rfData = await response.json();

    // Random predictions for other 8 models
    const randomLevel = () => Math.floor(Math.random() * 6) + 1 as 1 | 2 | 3 | 4 | 5 | 6;
    const otherModels = [
      "XGBoost Classifier",
      "SVM Classifier",
      "GNN Classifier",
      "BiLSTM Model",
      "TextCNN Classifier",
      "AraBERTv2 Classifier",
      "CAMeLBERT-mix Classifier",
      "CAMeLBERT-MSA Classifier",
    ].map(name => ({
      modelName: name,
      level: randomLevel(),
      confidence: +(Math.random() * 0.2 + 0.75).toFixed(2),
    }));

    const allPredictions = [...rfData.predictions, ...otherModels];

    // Hard vote: majority level
    const levels = allPredictions.map(p => p.level);
    const hardVote = levels.sort((a, b) =>
      levels.filter(v => v === a).length - levels.filter(v => v === b).length
    ).pop()!;

    return { predictions: allPredictions, hardVote };
  } catch (error) {
    throw error;
  }
};

const Index = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [classifiedWord, setClassifiedWord] = useState("");
  const { toast } = useToast();

  const handleClassify = async (word: string) => {
    setIsLoading(true);
    setResults(null);
    try {
      const data = await classifyWord(word);
      setResults(data);
      setClassifiedWord(word);
    } catch (error) {
      toast({
        title: "خطأ في التصنيف",
        description: "حدث خطأ أثناء تصنيف الكلمة. يرجى المحاولة مرة أخرى.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background" dir="rtl">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-center">
            <h1 className="font-bold text-primary drop-shadow-lg font-serif text-6xl">
              بَيِّنْ
            </h1>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        {/* Hero Section */}
        <div className="text-center mb-16 space-y-4">
          <h2 className="text-4xl md:text-5xl font-bold text-foreground leading-tight">
            تصنيف النصوص العربية حسب المستوى التعليمي
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            استخدام نموذج Random Forest مع توقعات عشوائية لبقية النماذج لتصنيف المستوى التعليمي للكلمة.
          </p>
        </div>

        {/* Input Section */}
        <div className="mb-16">
          <ClassificationInput onClassify={handleClassify} isLoading={isLoading} />
        </div>

        {/* Loading State */}
        {isLoading && (
          <div className="text-center py-16">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-primary border-t-transparent"></div>
            <p className="mt-4 text-muted-foreground">جارٍ التصنيف...</p>
          </div>
        )}

        {/* Results Section */}
        {results && !isLoading && (
          <ResultsDisplay
            word={classifiedWord}
            predictions={results.predictions.map((p: any) => ({
              ...p,
              color: levelColors[p.level] || "#000000",
            }))}
            hardVote={results.hardVote}
          />
        )}

        {/* Info Section */}
        {!results && !isLoading && (
          <div className="max-w-4xl mx-auto mt-16 p-8 bg-card border border-border rounded-lg">
            <h3 className="text-2xl font-bold text-foreground mb-4 text-center">
              حول نظام التصنيف
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-muted-foreground">
              <div>
                <h4 className="font-semibold text-foreground mb-2">المستويات التعليمية:</h4>
                <ul className="space-y-2 text-sm">
                  <li>• المستوى 1: مرحلة ما قبل المدرسة والصف 1-2</li>
                  <li>• المستوى 2: الصف 3-4</li>
                  <li>• المستوى 3: الصف 5-6</li>
                  <li>• المستوى 4: الصف 7-9</li>
                  <li>• المستوى 5: الصف 10-12</li>
                  <li>• المستوى 6: الجامعة</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-foreground mb-2">كيف يعمل النظام:</h4>
                <p className="text-sm">
                  يستخدم نظامنا نموذج Random Forest لتحليل وتصنيف النصوص، بينما يتم توليد نتائج عشوائية لبقية النماذج لتوضيح واجهة الاستخدام.
                </p>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-20 py-8 bg-card/30 text-center text-muted-foreground">
        © 2025 بَيِّنْ - نظام تصنيف الكلمات بالتعلم الآلي
      </footer>
    </div>
  );
};

export default Index;
