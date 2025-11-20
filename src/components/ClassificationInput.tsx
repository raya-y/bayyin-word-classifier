import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Search } from "lucide-react";

interface ClassificationInputProps {
  onClassify: (word: string) => void;
  isLoading: boolean;
}

export const ClassificationInput = ({ onClassify, isLoading }: ClassificationInputProps) => {
  const [word, setWord] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (word.trim()) {
      onClassify(word.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-2xl mx-auto">
      <div className="flex gap-3 items-center">
        <div className="relative flex-1">
          <Input
            type="text"
            value={word}
            onChange={(e) => setWord(e.target.value)}
            placeholder="أدخل كلمة للتصنيف... Enter a word to classify..."
            className="h-14 text-lg px-6 pr-12 bg-card border-2 border-border focus:border-primary transition-all duration-300 text-foreground placeholder:text-muted-foreground"
            disabled={isLoading}
          />
          <Search className="absolute right-4 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
        </div>
        <Button
          type="submit"
          disabled={!word.trim() || isLoading}
          size="lg"
          className="h-14 px-8 bg-primary hover:bg-primary/90 text-primary-foreground font-semibold text-lg shadow-lg hover:shadow-xl transition-all duration-300 disabled:opacity-50"
        >
          {isLoading ? "جارٍ التصنيف..." : "تصنيف"}
        </Button>
      </div>
    </form>
  );
};
