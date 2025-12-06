import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Filter, ChevronDown, X } from 'lucide-react';

interface AttributionFiltersProps {
  onFiltersChange: (filters: AttributionFilters) => void;
  totalCount: number;
  filteredCount: number;
}

export interface AttributionFilters {
  confidenceRange: [number, number];
  attributionTypes: string[];
  sourceTypes: string[];
  sortBy: 'confidence' | 'created_at' | 'similarity';
  sortOrder: 'asc' | 'desc';
}

const defaultFilters: AttributionFilters = {
  confidenceRange: [0, 100],
  attributionTypes: [],
  sourceTypes: [],
  sortBy: 'confidence',
  sortOrder: 'desc'
};

export default function AttributionFilters({ onFiltersChange, totalCount, filteredCount }: AttributionFiltersProps) {
  const [filters, setFilters] = useState<AttributionFilters>(defaultFilters);
  const [isOpen, setIsOpen] = useState(false);

  const updateFilters = (newFilters: Partial<AttributionFilters>) => {
    const updated = { ...filters, ...newFilters };
    setFilters(updated);
    onFiltersChange(updated);
  };

  const clearFilters = () => {
    setFilters(defaultFilters);
    onFiltersChange(defaultFilters);
  };

  const hasActiveFilters = () => {
    return (
      filters.confidenceRange[0] > 0 ||
      filters.confidenceRange[1] < 100 ||
      filters.attributionTypes.length > 0 ||
      filters.sourceTypes.length > 0
    );
  };

  const toggleArrayFilter = (value: string, array: string[], key: keyof Pick<AttributionFilters, 'attributionTypes' | 'sourceTypes'>) => {
    const updated = array.includes(value)
      ? array.filter(v => v !== value)
      : [...array, value];
    updateFilters({ [key]: updated });
  };

  return (
    <Card className="mb-6">
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CollapsibleTrigger asChild>
          <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2 text-lg">
                <Filter className="h-4 w-4" />
                Filters & Sorting
                {hasActiveFilters() && (
                  <Badge variant="secondary" className="ml-2">
                    {filteredCount}/{totalCount}
                  </Badge>
                )}
              </CardTitle>
              <div className="flex items-center gap-2">
                {hasActiveFilters() && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      clearFilters();
                    }}
                    className="h-7 px-2"
                  >
                    <X className="h-3 w-3 mr-1" />
                    Clear
                  </Button>
                )}
                <ChevronDown className={`h-4 w-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
              </div>
            </div>
          </CardHeader>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {/* Confidence Range */}
              <div className="space-y-3">
                <Label className="text-sm font-medium">Confidence Range</Label>
                <div className="px-2">
                  <Slider
                    value={filters.confidenceRange}
                    onValueChange={(value) => updateFilters({ confidenceRange: value as [number, number] })}
                    max={100}
                    min={0}
                    step={5}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground mt-1">
                    <span>{filters.confidenceRange[0]}%</span>
                    <span>{filters.confidenceRange[1]}%</span>
                  </div>
                </div>
              </div>

              {/* Attribution Types */}
              <div className="space-y-3">
                <Label className="text-sm font-medium">Attribution Type</Label>
                <div className="flex flex-wrap gap-2">
                  {['direct', 'paraphrase', 'summary', 'conceptual'].map((type) => (
                    <Badge
                      key={type}
                      variant={filters.attributionTypes.includes(type) ? 'default' : 'outline'}
                      className="cursor-pointer hover:opacity-80 transition-opacity capitalize"
                      onClick={() => toggleArrayFilter(type, filters.attributionTypes, 'attributionTypes')}
                    >
                      {type}
                    </Badge>
                  ))}
                </div>
              </div>

              {/* Source Types */}
              <div className="space-y-3">
                <Label className="text-sm font-medium">Source Type</Label>
                <div className="flex flex-wrap gap-2">
                  {['academic', 'web', 'book', 'article', 'dataset'].map((type) => (
                    <Badge
                      key={type}
                      variant={filters.sourceTypes.includes(type) ? 'default' : 'outline'}
                      className="cursor-pointer hover:opacity-80 transition-opacity capitalize"
                      onClick={() => toggleArrayFilter(type, filters.sourceTypes, 'sourceTypes')}
                    >
                      {type}
                    </Badge>
                  ))}
                </div>
              </div>

              {/* Sort Options */}
              <div className="space-y-3">
                <Label className="text-sm font-medium">Sort By</Label>
                <div className="space-y-2">
                  <Select
                    value={filters.sortBy}
                    onValueChange={(value) => updateFilters({ sortBy: value as AttributionFilters['sortBy'] })}
                  >
                    <SelectTrigger className="h-9">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="confidence">Confidence Score</SelectItem>
                      <SelectItem value="similarity">Similarity Score</SelectItem>
                      <SelectItem value="created_at">Date Created</SelectItem>
                    </SelectContent>
                  </Select>
                  <Select
                    value={filters.sortOrder}
                    onValueChange={(value) => updateFilters({ sortOrder: value as AttributionFilters['sortOrder'] })}
                  >
                    <SelectTrigger className="h-9">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="desc">Highest First</SelectItem>
                      <SelectItem value="asc">Lowest First</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>

            {/* Quick Filter Presets */}
            <div className="space-y-2">
              <Label className="text-sm font-medium">Quick Filters</Label>
              <div className="flex flex-wrap gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => updateFilters({ confidenceRange: [70, 100], sortBy: 'confidence' })}
                  className="h-8"
                >
                  High Risk Only
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => updateFilters({ attributionTypes: ['direct'], sortBy: 'similarity' })}
                  className="h-8"
                >
                  Direct Matches
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => updateFilters({ sourceTypes: ['academic'], sortBy: 'confidence' })}
                  className="h-8"
                >
                  Academic Sources
                </Button>
              </div>
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}