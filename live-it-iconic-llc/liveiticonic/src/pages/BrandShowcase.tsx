import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import SEO from '@/components/SEO';
import {
  ElegantFlamingo,
  MajesticPelican,
  CaribbeanFrigateBird,
  TropicalTanager,
} from '@/components/logo';
import { Home, Search, User, Settings, Menu, ShoppingCart, Heart, Bell, Zap } from 'lucide-react';

// ===== HELPER TYPES =====
type BackgroundType = 'dark' | 'light' | 'neon';

interface IconDefinition {
  Icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
  label: string;
}

interface FaviconItem {
  Component: React.ComponentType<React.SVGProps<SVGSVGElement>>;
  name: string;
  isRecommended?: boolean;
}

// ===== STYLE LOOKUP OBJECTS =====
const BACKGROUND_STYLES: Record<BackgroundType, string> = {
  dark: 'bg-lii-bg border-lii-gold/20',
  light: 'bg-lii-cloud border-lii-charcoal/20',
  neon: 'bg-gradient-to-br from-purple-900 via-pink-500 to-cyan-400 border-transparent',
};

const LOGO_COLOR_MAP: Record<BackgroundType, string> = {
  dark: 'text-lii-gold',
  light: 'text-lii-bg',
  neon: 'text-white',
};

const HEADING_COLOR_MAP: Record<BackgroundType, string> = {
  dark: 'text-lii-cloud',
  light: 'text-lii-bg',
  neon: 'text-lii-cloud',
};

const TEXT_COLOR_MAP: Record<BackgroundType, string> = {
  dark: 'text-lii-ash',
  light: 'text-lii-charcoal',
  neon: 'text-lii-ash',
};

// ===== HELPER FUNCTIONS =====
const getBackgroundStyle = (bg: BackgroundType): string => BACKGROUND_STYLES[bg];

const getLogoColor = (bg: BackgroundType): string => LOGO_COLOR_MAP[bg];

const getHeadingColor = (bg: BackgroundType): string => HEADING_COLOR_MAP[bg];

const getTextColor = (bg: BackgroundType): string => TEXT_COLOR_MAP[bg];

const ICON_DEFINITIONS: IconDefinition[] = [
  { Icon: Home, label: 'Home' },
  { Icon: Search, label: 'Search' },
  { Icon: User, label: 'User' },
  { Icon: Settings, label: 'Settings' },
  { Icon: ShoppingCart, label: 'Cart' },
  { Icon: Heart, label: 'Favorite' },
  { Icon: Menu, label: 'Menu' },
  { Icon: Bell, label: 'Notify' },
];

// ===== ICON GRID CONFIG =====
const ICON_GRID_CONFIG: Record<
  'solid' | 'outline' | 'dual-tone',
  { title: string; colorClass: string; shouldFill: boolean }
> = {
  solid: { title: 'Solid / Filled', colorClass: 'text-lii-cloud', shouldFill: true },
  outline: { title: 'Outline / Stroke', colorClass: 'text-lii-cloud', shouldFill: false },
  'dual-tone': { title: 'Dual-Tone / Accent', colorClass: 'text-lii-gold', shouldFill: true },
};

// ===== REUSABLE COMPONENTS =====
interface IconGridProps {
  style: 'solid' | 'outline' | 'dual-tone';
  description: string;
}

const IconGrid: React.FC<IconGridProps> = ({ style, description }) => {
  const config = ICON_GRID_CONFIG[style];

  return (
    <Card className="bg-lii-charcoal/30 border-lii-gold/20 p-6">
      <h3 className="text-lii-gold font-semibold mb-2 text-sm uppercase tracking-wider">
        {config.title}
      </h3>
      <p className="text-lii-ash text-sm mb-4">{description}</p>
      <div className="grid grid-cols-4 gap-4">
        {ICON_DEFINITIONS.map(({ Icon, label }) => (
          <div key={label} className="flex flex-col items-center gap-2">
            <Icon
              className={`w-8 h-8 ${config.colorClass}`}
              fill={config.shouldFill ? 'currentColor' : 'none'}
            />
            <span className="text-xs text-lii-ash">{label}</span>
          </div>
        ))}
      </div>
    </Card>
  );
};

interface LogoCardProps {
  Component: React.ComponentType<React.SVGProps<SVGSVGElement> & { size?: number }>;
  name: string;
  description: string;
  bg: BackgroundType;
}

const LogoCard: React.FC<LogoCardProps> = ({ Component, name, description, bg }) => {
  const bgStyle = getBackgroundStyle(bg);
  const logoColor = getLogoColor(bg);
  const headingColor = getHeadingColor(bg);
  const textColor = getTextColor(bg);

  return (
    <Card className={`p-10 transition-all duration-300 ${bgStyle}`}>
      <div className="flex flex-col items-center">
        <div className="mb-8">
          <Component size={140} className={logoColor} />
        </div>
        <h3 className={`font-display text-2xl mb-3 ${headingColor}`}>{name}</h3>
        <p className={`text-base text-center ${textColor}`}>{description}</p>
      </div>
    </Card>
  );
};

interface FaviconDisplayProps {
  Component: React.ComponentType<React.SVGProps<SVGSVGElement> & { size?: number }>;
  name: string;
  isRecommended?: boolean;
}

const FaviconDisplay: React.FC<FaviconDisplayProps> = ({ Component, name, isRecommended }) => {
  return (
    <div className="text-center">
      <div className="bg-lii-bg border-2 border-lii-gold/30 rounded-lg p-4 mb-4 inline-block">
        <Component size={64} className="text-lii-gold" />
      </div>
      <p className="text-lii-cloud text-sm font-semibold mb-2">{name}</p>
      <div className="flex gap-2 justify-center mb-2">
        <div className="w-4 h-4 bg-lii-bg border border-lii-gold/30 rounded flex items-center justify-center">
          <Component size={12} className="text-lii-gold" />
        </div>
        <div className="w-8 h-8 bg-lii-bg border border-lii-gold/30 rounded flex items-center justify-center">
          <Component size={24} className="text-lii-gold" />
        </div>
      </div>
      {isRecommended && (
        <Badge className="bg-lii-gold/20 text-lii-gold border-lii-gold/40">
          My pick
        </Badge>
      )}
    </div>
  );
};

const BrandShowcase = () => {
  const [selectedTheme, setSelectedTheme] = useState<string>('minimal');
  const [selectedBg, setSelectedBg] = useState<BackgroundType>('dark');

  // 5 Theme Concepts
  const themes = [
    {
      id: 'minimal',
      name: 'Ultra Modern Minimal',
      description: 'Clean lines, maximum whitespace, monochromatic with gold accents',
      colors: { bg: '#FFFFFF', text: '#0B0B0C', accent: '#C1A060', border: '#E6E9EF' },
      fonts: { display: 'Inter', body: 'Inter' },
      buttons: 'Sharp rectangles, minimal padding, subtle hover states',
      nav: 'Floating pill with blur, ultra-thin borders',
      headings: 'Tight tracking, no serifs, weight 300-600',
      vibe: 'Apple-inspired, luxury minimalism, breathing room',
    },
    {
      id: 'retro90s',
      name: 'Retro 90s',
      description: 'Bold typography, bright colors, geometric patterns, nostalgic vibes',
      colors: { bg: '#FF6B9D', text: '#FFED00', accent: '#00D9FF', border: '#6E44FF' },
      fonts: { display: 'Impact', body: 'Arial' },
      buttons: 'Chunky, beveled edges, neon outlines',
      nav: 'Horizontal bar with gradient, jagged dividers',
      headings: 'ALL CAPS, thick strokes, shadow effects',
      vibe: 'Fresh Prince, MTV, arcade games, maximalist energy',
    },
    {
      id: 'vaporwave',
      name: 'Vaporwave Gradient',
      description: 'Dreamy gradients, pink/purple/cyan, glitch aesthetics, futuristic',
      colors: { bg: '#1a0033', text: '#FF6FF2', accent: '#00FFF7', border: '#9D4EDD' },
      fonts: { display: 'Orbitron', body: 'Roboto' },
      buttons: 'Rounded, gradient fills, neon glow on hover',
      nav: 'Transparent with heavy blur, gradient border',
      headings: 'Italic, condensed, chromatic aberration effect',
      vibe: 'Blade Runner, cyberpunk, A E S T H E T I C',
    },
    {
      id: 'nature',
      name: 'Nature Cozy',
      description: 'Earthy tones, organic shapes, warm and inviting',
      colors: { bg: '#F5F1E8', text: '#2A3B2E', accent: '#8B6F47', border: '#A8B89F' },
      fonts: { display: 'Merriweather', body: 'Source Sans Pro' },
      buttons: 'Rounded corners, matte finish, gentle shadows',
      nav: 'Sticky top, subtle shadow, natural texture',
      headings: 'Serif, warm weight, generous line height',
      vibe: 'Coffee shop, Sunday morning, handcrafted feel',
    },
    {
      id: 'playful',
      name: 'Playful Cartoonish',
      description: 'Bold colors, fun shapes, animated elements, friendly',
      colors: { bg: '#FFF9E5', text: '#2D2D2D', accent: '#FF5C5C', border: '#FFD93D' },
      fonts: { display: 'Fredoka One', body: 'Nunito' },
      buttons: 'Pill-shaped, bouncy animations, colorful shadows',
      nav: 'Wave pattern, illustrated icons, cheerful spacing',
      headings: 'Rounded, heavy weight, playful kerning',
      vibe: 'Nintendo, kids toys, approachable and fun',
    },
  ];

  const currentTheme = themes.find(t => t.id === selectedTheme) || themes[0];

  return (
    <>
      <SEO
        title="Brand Showcase — Live It Iconic"
        description="Explore theme concepts, logo variations, icon styles, and merchandise designs for Live It Iconic"
      />

      <div className="min-h-screen bg-lii-bg pt-32 pb-20">
        <div className="container mx-auto px-4 sm:px-6 max-w-7xl">
          {/* Header */}
          <div className="text-center mb-16">
            <h1 className="font-display text-5xl sm:text-6xl text-lii-cloud mb-4 tracking-tight">
              Rio — Here's What I Put Together
            </h1>
            <p className="text-lii-ash text-xl max-w-4xl mx-auto leading-relaxed">
              Five complete website themes, bird logo variations, icon sets in three styles, and
              merch mockups. Each one's got a totally different vibe. Pick what feels right and
              we'll build it out.
            </p>
          </div>

          <Tabs defaultValue="themes" className="w-full">
            <TabsList className="grid w-full grid-cols-5 mb-12 bg-lii-charcoal/50">
              <TabsTrigger value="themes">Themes</TabsTrigger>
              <TabsTrigger value="logos">Logos</TabsTrigger>
              <TabsTrigger value="icons">Icons</TabsTrigger>
              <TabsTrigger value="merch">Merch</TabsTrigger>
              <TabsTrigger value="favicon">Favicon</TabsTrigger>
            </TabsList>

            {/* === THEMES TAB === */}
            <TabsContent value="themes" className="space-y-12">
              {/* Theme Selector */}
              <div className="flex flex-wrap gap-3 justify-center mb-8">
                {themes.map(theme => (
                  <Button
                    key={theme.id}
                    onClick={() => setSelectedTheme(theme.id)}
                    variant={selectedTheme === theme.id ? 'default' : 'outline'}
                    className="min-w-[140px]"
                  >
                    {theme.name}
                  </Button>
                ))}
              </div>

              {/* Current Theme Details */}
              <Card className="bg-lii-charcoal/30 border-lii-gold/20 p-8">
                <div className="grid md:grid-cols-2 gap-8">
                  <div>
                    <h2 className="font-display text-3xl text-lii-cloud mb-3">
                      {currentTheme.name}
                    </h2>
                    <p className="text-lii-ash text-lg mb-8 leading-relaxed">
                      {currentTheme.description}
                    </p>

                    <div className="space-y-5">
                      <div>
                        <h3 className="text-lii-gold font-semibold mb-2 text-sm uppercase tracking-wider">
                          The Vibe
                        </h3>
                        <p className="text-lii-cloud text-base leading-relaxed">
                          {currentTheme.vibe}
                        </p>
                      </div>
                      <div>
                        <h3 className="text-lii-gold font-semibold mb-2 text-sm uppercase tracking-wider">
                          Typography
                        </h3>
                        <p className="text-lii-cloud text-base">
                          Headlines in {currentTheme.fonts.display}, body copy in{' '}
                          {currentTheme.fonts.body}
                        </p>
                      </div>
                      <div>
                        <h3 className="text-lii-gold font-semibold mb-2 text-sm uppercase tracking-wider">
                          Button Style
                        </h3>
                        <p className="text-lii-cloud text-base">{currentTheme.buttons}</p>
                      </div>
                      <div>
                        <h3 className="text-lii-gold font-semibold mb-2 text-sm uppercase tracking-wider">
                          Nav Treatment
                        </h3>
                        <p className="text-lii-cloud text-base">{currentTheme.nav}</p>
                      </div>
                      <div>
                        <h3 className="text-lii-gold font-semibold mb-2 text-sm uppercase tracking-wider">
                          Heading Style
                        </h3>
                        <p className="text-lii-cloud text-base">{currentTheme.headings}</p>
                      </div>
                    </div>
                  </div>

                  {/* Color Palette */}
                  <div>
                    <h3 className="text-lii-gold font-semibold mb-4">Color Palette</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <div
                          className="w-full h-24 rounded-lg border-2 border-lii-gold/30 mb-2"
                          style={{ backgroundColor: currentTheme.colors.bg }}
                        ></div>
                        <p className="text-sm text-lii-ash">Background</p>
                        <p className="text-xs text-lii-cloud font-mono">{currentTheme.colors.bg}</p>
                      </div>
                      <div>
                        <div
                          className="w-full h-24 rounded-lg border-2 border-lii-gold/30 mb-2"
                          style={{ backgroundColor: currentTheme.colors.text }}
                        ></div>
                        <p className="text-sm text-lii-ash">Text</p>
                        <p className="text-xs text-lii-cloud font-mono">
                          {currentTheme.colors.text}
                        </p>
                      </div>
                      <div>
                        <div
                          className="w-full h-24 rounded-lg border-2 border-lii-gold/30 mb-2"
                          style={{ backgroundColor: currentTheme.colors.accent }}
                        ></div>
                        <p className="text-sm text-lii-ash">Accent</p>
                        <p className="text-xs text-lii-cloud font-mono">
                          {currentTheme.colors.accent}
                        </p>
                      </div>
                      <div>
                        <div
                          className="w-full h-24 rounded-lg border-2 border-lii-gold/30 mb-2"
                          style={{ backgroundColor: currentTheme.colors.border }}
                        ></div>
                        <p className="text-sm text-lii-ash">Border</p>
                        <p className="text-xs text-lii-cloud font-mono">
                          {currentTheme.colors.border}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>

              {/* Mockup Preview */}
              <Card className="bg-lii-charcoal/30 border-lii-gold/20 p-8">
                <h3 className="text-lii-gold font-semibold mb-2 text-sm uppercase tracking-wider">
                  How It Looks Live
                </h3>
                <p className="text-lii-ash mb-6">
                  Quick hero section mockup with this theme applied
                </p>
                <div
                  className="rounded-xl border-2 p-8 min-h-[400px]"
                  style={{
                    backgroundColor: currentTheme.colors.bg,
                    borderColor: currentTheme.colors.border,
                  }}
                >
                  <h1
                    className="text-4xl font-bold mb-4"
                    style={{
                      color: currentTheme.colors.text,
                      fontFamily: currentTheme.fonts.display,
                    }}
                  >
                    Live It Iconic
                  </h1>
                  <p
                    className="text-lg mb-6"
                    style={{
                      color: currentTheme.colors.text,
                      fontFamily: currentTheme.fonts.body,
                      opacity: 0.8,
                    }}
                  >
                    Statement pieces for bold days
                  </p>
                  <button
                    className="px-8 py-3 rounded-lg font-semibold transition-all"
                    style={{
                      backgroundColor: currentTheme.colors.accent,
                      color: currentTheme.colors.bg,
                    }}
                  >
                    Shop Now
                  </button>
                </div>
              </Card>
            </TabsContent>

            {/* === LOGOS TAB === */}
            <TabsContent value="logos" className="space-y-8">
              <div className="text-center mb-8">
                <h2 className="font-display text-3xl text-lii-cloud mb-3">The Bird Logos</h2>
                <p className="text-lii-ash text-lg max-w-2xl mx-auto">
                  Four Caribbean birds, each with a different character. Test them on dark, light,
                  and neon backgrounds to see what works best.
                </p>
              </div>

              {/* Background Selector */}
              <div className="flex gap-3 justify-center mb-8">
                <Button
                  onClick={() => setSelectedBg('dark')}
                  variant={selectedBg === 'dark' ? 'default' : 'outline'}
                >
                  Dark Background
                </Button>
                <Button
                  onClick={() => setSelectedBg('light')}
                  variant={selectedBg === 'light' ? 'default' : 'outline'}
                >
                  Light Background
                </Button>
                <Button
                  onClick={() => setSelectedBg('neon')}
                  variant={selectedBg === 'neon' ? 'default' : 'outline'}
                >
                  Neon Gradient
                </Button>
              </div>

              {/* Logo Variations */}
              <div className="grid md:grid-cols-2 gap-8">
                <LogoCard
                  Component={ElegantFlamingo}
                  name="Elegant Flamingo"
                  description="Grace and refinement — perfect for upscale minimal look"
                  bg={selectedBg}
                />
                <LogoCard
                  Component={MajesticPelican}
                  name="Majestic Pelican"
                  description="Strength and endurance — bold statement piece"
                  bg={selectedBg}
                />
                <LogoCard
                  Component={CaribbeanFrigateBird}
                  name="Caribbean Frigate"
                  description="Freedom and power — dynamic energy"
                  bg={selectedBg}
                />
                <LogoCard
                  Component={TropicalTanager}
                  name="Tropical Tanager"
                  description="Compact and vibrant — works great small"
                  bg={selectedBg}
                />
              </div>
            </TabsContent>

            {/* === ICONS TAB === */}
            <TabsContent value="icons" className="space-y-8">
              <div className="text-center mb-8">
                <h2 className="font-display text-3xl text-lii-cloud mb-3">Icon Library</h2>
                <p className="text-lii-ash text-lg max-w-2xl mx-auto">
                  Three styles: solid (filled), outline (strokes only), and dual-tone. Pick one
                  style and stick with it across the whole site for consistency.
                </p>
              </div>

              <div className="grid md:grid-cols-3 gap-8">
                <IconGrid
                  style="solid"
                  description="Bold and modern, great for primary actions"
                />
                <IconGrid
                  style="outline"
                  description="Clean and minimal, works on any background"
                />
                <IconGrid
                  style="dual-tone"
                  description="Gold fill, perfect for premium feel"
                />
              </div>

              {/* Animated Icons Note */}
              <Card className="bg-lii-charcoal/30 border-lii-gold/20 p-6">
                <h3 className="text-lii-gold font-semibold mb-2 text-sm uppercase tracking-wider">
                  Animated Micro-interactions
                </h3>
                <p className="text-lii-ash leading-relaxed">
                  We can add subtle animations: bird flapping wings on hover, heart pulse when
                  clicked, bell ring, cart bounce. All doable with CSS or Lottie files. Let me know
                  if you want these for launch or save them for later.
                </p>
              </Card>
            </TabsContent>

            {/* === MERCH TAB === */}
            <TabsContent value="merch" className="space-y-8">
              <div className="text-center mb-8">
                <h2 className="font-display text-3xl text-lii-cloud mb-3">Merch Mockups</h2>
                <p className="text-lii-ash text-lg max-w-2xl mx-auto">
                  Hoodies, hats, stickers, and a deconstructed minimal treatment. These are
                  conceptual—once you pick a logo, I can generate proper product mockups.
                </p>
              </div>

              <div className="grid md:grid-cols-2 gap-8">
                {/* Hoodie Mockup */}
                <Card className="bg-lii-charcoal/30 border-lii-gold/20 p-6">
                  <h3 className="text-lii-gold font-semibold mb-2 text-sm uppercase tracking-wider">
                    Hoodie — Chest Logo
                  </h3>
                  <p className="text-lii-ash text-sm mb-4">
                    Bold front placement, clean and iconic
                  </p>
                  <div className="bg-lii-bg rounded-lg p-8 min-h-[300px] flex items-center justify-center border border-lii-gold/20">
                    <div className="text-center">
                      <ElegantFlamingo size={100} className="text-lii-gold mx-auto mb-4" />
                      <p className="text-lii-cloud font-display text-lg tracking-wider">
                        LIVE IT ICONIC
                      </p>
                      <Badge className="mt-3 bg-lii-gold/20 text-lii-gold border-lii-gold/40">
                        Black / Tan / Charcoal
                      </Badge>
                    </div>
                  </div>
                </Card>

                {/* Dad Hat Mockup */}
                <Card className="bg-lii-charcoal/30 border-lii-gold/20 p-6">
                  <h3 className="text-lii-gold font-semibold mb-2 text-sm uppercase tracking-wider">
                    Dad Hat — Embroidered Side
                  </h3>
                  <p className="text-lii-ash text-sm mb-4">Subtle placement, premium stitching</p>
                  <div className="bg-lii-cloud rounded-lg p-8 min-h-[300px] flex items-center justify-center border border-lii-gold/20">
                    <div className="text-center">
                      <ElegantFlamingo size={64} className="text-lii-bg mx-auto mb-4" />
                      <p className="text-lii-bg font-display text-sm tracking-wide">
                        Left panel placement
                      </p>
                      <Badge className="mt-3 bg-lii-bg/10 text-lii-bg border-lii-bg/40">
                        White / Navy / Stone
                      </Badge>
                    </div>
                  </div>
                </Card>

                {/* Sticker Sheet */}
                <Card className="bg-lii-charcoal/30 border-lii-gold/20 p-6">
                  <h3 className="text-lii-gold font-semibold mb-2 text-sm uppercase tracking-wider">
                    Sticker Sheet
                  </h3>
                  <p className="text-lii-ash text-sm mb-4">All four birds + wordmark + icons</p>
                  <div className="bg-lii-bg rounded-lg p-8 min-h-[300px] border border-lii-gold/20">
                    <div className="grid grid-cols-3 gap-6">
                      <ElegantFlamingo size={60} className="text-lii-gold" />
                      <MajesticPelican size={60} className="text-lii-gold" />
                      <CaribbeanFrigateBird size={60} className="text-lii-gold" />
                      <TropicalTanager size={60} className="text-lii-gold" />
                      <div className="flex items-center justify-center">
                        <p className="text-lii-gold font-display text-xs text-center tracking-wider">
                          LIVE IT
                          <br />
                          ICONIC
                        </p>
                      </div>
                      <div className="flex items-center justify-center">
                        <Zap className="w-12 h-12 text-lii-gold" fill="currentColor" />
                      </div>
                    </div>
                  </div>
                </Card>

                {/* Deconstructed Look */}
                <Card className="bg-lii-charcoal/30 border-lii-gold/20 p-6">
                  <h3 className="text-lii-gold font-semibold mb-2 text-sm uppercase tracking-wider">
                    Deconstructed / Minimal
                  </h3>
                  <p className="text-lii-ash text-sm mb-4">Split branding across seams and tags</p>
                  <div className="bg-lii-bg rounded-lg p-8 min-h-[300px] flex flex-col justify-between border border-lii-gold/20">
                    <div className="flex justify-between items-start">
                      <ElegantFlamingo size={40} className="text-lii-gold" />
                      <p className="text-lii-cloud font-display text-sm tracking-[0.3em]">
                        LIVE IT
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-lii-ash text-xs italic mb-2">
                        Small tag on sleeve or inside hem
                      </p>
                      <Badge className="bg-lii-gold/20 text-lii-gold border-lii-gold/40">
                        Understated luxury
                      </Badge>
                    </div>
                  </div>
                </Card>
              </div>
            </TabsContent>

            {/* === FAVICON TAB === */}
            <TabsContent value="favicon" className="space-y-8">
              <div className="text-center mb-8">
                <h2 className="font-display text-3xl text-lii-cloud mb-3">Favicon Concepts</h2>
                <p className="text-lii-ash text-lg max-w-2xl mx-auto">
                  Just the bird, no text—stays crisp at tiny sizes (16x16 up to 64x64). Pick the one
                  that reads best when it's postage-stamp small.
                </p>
              </div>

              <Card className="bg-lii-charcoal/30 border-lii-gold/20 p-8">
                <div className="grid md:grid-cols-4 gap-8">
                  <FaviconDisplay
                    Component={ElegantFlamingo}
                    name="Elegant Flamingo"
                    isRecommended
                  />
                  <FaviconDisplay Component={MajesticPelican} name="Majestic Pelican" />
                  <FaviconDisplay
                    Component={CaribbeanFrigateBird}
                    name="Frigate Bird"
                  />
                  <FaviconDisplay Component={TropicalTanager} name="Tropical Tanager" />
                </div>
              </Card>
            </TabsContent>
          </Tabs>

          {/* Call to Action */}
          <Card className="bg-lii-charcoal/30 border-lii-gold/20 p-8 mt-16 text-center">
            <h2 className="font-display text-3xl text-lii-cloud mb-4">
              So Rio, What Do You Think?
            </h2>
            <p className="text-lii-ash text-lg mb-8 max-w-3xl mx-auto leading-relaxed">
              Take your time going through each theme, logo, and mockup. When you've got a favorite
              (or two), let's chat about it. I can build out whichever direction feels right—or mix
              and match if you want elements from different themes.
            </p>
            <div className="flex gap-4 justify-center flex-wrap">
              <Button variant="default" size="lg" className="min-w-[200px]">
                Let's Discuss
              </Button>
              <Button variant="outline" size="lg" className="min-w-[200px]">
                Save for Later
              </Button>
            </div>
          </Card>
        </div>
      </div>
    </>
  );
};

export default BrandShowcase;
