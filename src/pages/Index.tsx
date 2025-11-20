import { useState } from "react";
import { ClassificationInput } from "@/components/ClassificationInput";
import { ResultsDisplay } from "@/components/ResultsDisplay";
import { useToast } from "@/hooks/use-toast";

// Placeholder function for backend integration
const classifyWord = async (word: string) => {
  // TODO: Replace with actual API call to Python backend (Flask/FastAPI)
  // Example: const response = await fetch('/api/classify', { method: 'POST', body: JSON.stringify({ word }) });

  // Simulated response for demo purposes
  await new Promise(resolve => setTimeout(resolve, 1000));
  const randomLevel = () => Math.floor(Math.random() * 6) + 1 as 1 | 2 | 3 | 4 | 5 | 6;
  return {
    predictions: [{
      modelName: "BERT Classifier",
      level: randomLevel(),
      confidence: 0.85
    }, {
      modelName: "LSTM Model",
      level: randomLevel(),
      confidence: 0.78
    }, {
      modelName: "CNN Classifier",
      level: randomLevel(),
      confidence: 0.92
    }, {
      modelName: "Transformer Model",
      level: randomLevel(),
      confidence: 0.88
    }, {
      modelName: "Random Forest",
      level: randomLevel(),
      confidence: 0.76
    }, {
      modelName: "SVM Classifier",
      level: randomLevel(),
      confidence: 0.81
    }, {
      modelName: "Naive Bayes",
      level: randomLevel(),
      confidence: 0.73
    }, {
      modelName: "Logistic Regression",
      level: randomLevel(),
      confidence: 0.79
    }, {
      modelName: "XGBoost",
      level: randomLevel(),
      confidence: 0.86
    }],
    hardVote: randomLevel()
  };
};
const Index = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [classifiedWord, setClassifiedWord] = useState("");
  const {
    toast
  } = useToast();
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
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };
  return <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-center">
            <h1 className="text-5xl font-bold text-primary drop-shadow-lg font-serif">
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
            Classify Arabic Words by
            <span className="text-primary block mt-2">Educational Level</span>
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Using 9 advanced ML and DL models to determine the appropriate educational level for any word
          </p>
        </div>

        {/* Input Section */}
        <div className="mb-16">
          <ClassificationInput onClassify={handleClassify} isLoading={isLoading} />
        </div>

        {/* Loading State */}
        {isLoading && <div className="text-center py-16">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-primary border-t-transparent"></div>
            <p className="mt-4 text-muted-foreground">Classifying word...</p>
          </div>}

        {/* Results Section */}
        {results && !isLoading && <ResultsDisplay word={classifiedWord} predictions={results.predictions} hardVote={results.hardVote} />}

        {/* Info Section */}
        {!results && !isLoading && <div className="max-w-4xl mx-auto mt-16 p-8 bg-card border border-border rounded-lg">
            <h3 className="text-2xl font-bold text-foreground mb-4 text-center">
              About the Classification System
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-muted-foreground">
              <div>
                <h4 className="font-semibold text-foreground mb-2">Educational Levels:</h4>
                <ul className="space-y-2 text-sm">
                  <li>• Level 1: Pre-school & Grades 1-2</li>
                  <li>• Level 2: Grades 3-4</li>
                  <li>• Level 3: Grades 5-6</li>
                  <li>• Level 4: Grades 7-9</li>
                  <li>• Level 5: Grades 10-12</li>
                  <li>• Level 6: University</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-foreground mb-2">How it works:</h4>
                <p className="text-sm">
                  Our system uses 9 different machine learning and deep learning models 
                  to analyze and classify words based on their complexity and appropriate 
                  educational level. The consensus result represents the most agreed-upon 
                  classification across all models.
                </p>
              </div>
            </div>
          </div>}
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-20 py-8 bg-card/30">
        <div className="container mx-auto px-4 text-center text-muted-foreground">
          <p>© 2025 Bayyin (بَيِّنْ) - ML Word Classification System</p>
        </div>
      </footer>
    </div>;
};
export default Index;