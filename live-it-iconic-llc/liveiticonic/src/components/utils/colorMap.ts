const COLOR_HEX_MAP: Record<string, string> = {
  Black: '#000',
  White: '#fff',
  Charcoal: '#374151',
  Navy: '#1e3a8a',
  Olive: '#65a30d',
  Forest: '#166534',
  Cream: '#fef3c7',
  Ash: '#f3f4f6',
  Stone: '#d1d5db',
  Obsidian: '#111827',
};

export const getColorHex = (colorName: string): string => {
  return COLOR_HEX_MAP[colorName] || '#6b7280';
};

export const getAvailabilityColor = (availability: string): string => {
  const colorMap: Record<string, string> = {
    'In Stock': 'text-green-300 bg-green-500/10',
    Limited: 'text-amber-300 bg-amber-500/10',
    'Pre-Order': 'text-blue-300 bg-blue-500/10',
  };
  return colorMap[availability] || 'text-foreground/60 bg-foreground/10';
};

export const getBirdType = (index: number): 'flamingo' | 'pelican' | 'frigate' | 'tanager' => {
  const types: ('flamingo' | 'pelican' | 'frigate' | 'tanager')[] = ['flamingo', 'pelican', 'frigate', 'tanager'];
  return types[index % 4];
};
