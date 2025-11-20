import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Search } from "lucide-react";
interface ClassificationInputProps {
  onClassify: (word: string) => void;
  isLoading: boolean;
}
export const ClassificationInput = ({
  onClassify,
  isLoading
}: ClassificationInputProps) => {
  const [word, setWord] = useState("");
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (word.trim()) {
      onClassify(word.trim());
    }
  };
  return <form onSubmit={handleSubmit} className="w-full max-w-2xl mx-auto">
      <div className="flex flex-col gap-3">
        <div className="relative">
          <Textarea value={word} onChange={e => setWord(e.target.value)} className="min-h-32 text-lg px-6 py-4 bg-card border-2 border-border focus:border-primary transition-all duration-300 text-foreground placeholder:text-muted-foreground resize-none" disabled={isLoading} placeholder="أدخل نصاً للتصنيف..." dir="rtl" />
        </div>
        <Button type="submit" disabled={!word.trim() || isLoading} size="lg" className="w-full h-14 bg-primary hover:bg-primary/90 text-primary-foreground font-semibold text-lg shadow-lg hover:shadow-xl transition-all duration-300 disabled:opacity-50">
          {isLoading ? "جارٍ التصنيف..." : "تصنيف"}
        </Button>
      </div>
    </form>;
};