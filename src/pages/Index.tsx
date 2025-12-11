import { useState } from "react";
import { ClassificationInput } from "@/components/ClassificationInput";
import { ResultsDisplay } from "@/components/ResultsDisplay";
import { useToast } from "@/hooks/use-toast";

// Backend API URL
// For development: http://localhost:8000
// For production: Hugging Face Spaces
const API_URL = import.meta.env.VITE_API_URL || 'https://raya-y-bayyin-backend.hf.space';
const classifyWord = async (word: string) => {
  try {
    const response = await fetch(`${API_URL}/classify`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        text: word
      })
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({
        detail: 'Classification failed'
      }));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }
    const data = await response.json();

    // Transform backend response to frontend format
    return {
      predictions: data.prediction_results.map((result: any) => ({
        modelName: result.model,
        level: parseInt(result.prediction.split(' ')[1]) as 1 | 2 | 3 | 4 | 5 | 6,
        confidence: result.confidence
      })),
      hardVote: data.ensemble_decision ? parseInt(data.ensemble_decision.split(' ')[1]) as 1 | 2 | 3 | 4 | 5 | 6 : 1
    };
  } catch (error) {
    // Re-throw to be handled by the component
    throw error;
  }
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
    } catch (error: any) {
      // Extract error message
      let errorMessage = error?.message || "حدث خطأ أثناء تصنيف الكلمة. يرجى المحاولة مرة أخرى.";

      // Check if it's a network error (backend not running)
      if (errorMessage.includes('Failed to fetch') || errorMessage.includes('NetworkError')) {
        errorMessage = "لا يمكن الاتصال بالخادم. يرجى التأكد من تشغيل الخادم على المنفذ 8000.";
      }

      // Check if it's a language detection error (non-Arabic text)
      const isLanguageError = errorMessage.toLowerCase().includes('not arabic') || errorMessage.toLowerCase().includes('language') || errorMessage.toLowerCase().includes('detected language');
      toast({
        title: isLanguageError ? "خطأ في اللغة" : "خطأ في التصنيف",
        description: isLanguageError ? "النص المدخل ليس بالعربية. يرجى إدخال نص عربي للتصنيف." : errorMessage,
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
            <h1 className="font-bold text-primary drop-shadow-lg font-serif text-6xl">
              بَيِّنْ
            </h1>
          </div>
          
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        {/* Hero Section */}
        <div className="text-center mb-16 space-y-4" dir="rtl">
          <h2 className="text-4xl md:text-5xl font-bold text-foreground leading-tight">منصة ذكية لتحليل الجمل العربية وتقدير مستواها التعليمي</h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">بين هو نظام ذكي يصنف الجمل العربية بدقة بحسب المستوى التعليمي باستخدام تقنيات حديثة في تعلم الآلة والتعلم العميق ونماذج المحولات.</p>
        </div>

        {/* Input Section */}
        <div className="mb-16">
          <ClassificationInput onClassify={handleClassify} isLoading={isLoading} />
        </div>

        {/* Loading State */}
        {isLoading && <div className="text-center py-16" dir="rtl">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-primary border-t-transparent"></div>
            <p className="mt-4 text-muted-foreground">جارٍ التصنيف...</p>
          </div>}

        {/* Results Section */}
        {results && !isLoading && <ResultsDisplay word={classifiedWord} predictions={results.predictions} hardVote={results.hardVote} />}

        {/* Info Section */}
        {!results && !isLoading && <div className="max-w-4xl mx-auto mt-16 p-8 bg-card border border-border rounded-lg" dir="rtl">
            <h3 className="text-2xl font-bold text-foreground mb-4 text-center">
              حول نظام التصنيف
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-muted-foreground">
              <div>
                <h4 className="font-semibold text-foreground mb-2">المستويات التعليمية:</h4>
                <ul className="space-y-2 text-sm">
                  <li>• المستوى 1: الصف 1-2</li>
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
                  يستخدم نظامنا 9 نماذج مختلفة من تعلم الآلة والتعلم العميق ونماذج المحولات لتحليل وتصنيف النصوص بناءً على تعقيدها والمستوى التعليمي المناسب. تمثل النتيجة النهائية التصنيف الأكثر اتفاقًا بين جميع النماذج.
                </p>
              </div>
            </div>
          </div>}
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-20 py-8 bg-card/30" dir="rtl">
        <div className="container mx-auto px-4 text-center text-muted-foreground">
          <p>© 2025 بَيِّنْ - نظام تصنيف الكلمات بالتعلم الآلي</p>
        </div>
      </footer>
    </div>;
};
export default Index;