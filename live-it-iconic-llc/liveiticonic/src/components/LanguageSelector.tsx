import { useTranslation } from 'react-i18next';
import { Globe } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Button } from '@/components/ui/button';

interface Language {
  code: string;
  name: string;
  flag: string;
}

const languages: Language[] = [
  { code: 'en', name: 'English', flag: '🇺🇸' },
  { code: 'fr', name: 'Français', flag: '🇫🇷' },
  { code: 'es', name: 'Español', flag: '🇪🇸' },
];

/**
 * Language Selector Component
 * Allows users to switch between available languages
 */
export function LanguageSelector() {
  const { i18n } = useTranslation();

  const currentLanguage = languages.find((lang) => lang.code === i18n.language);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="flex items-center gap-2"
          aria-label="Select language"
        >
          <Globe className="h-4 w-4" />
          <span className="hidden sm:inline text-sm font-medium">
            {currentLanguage?.flag} {i18n.language.toUpperCase()}
          </span>
          <span className="sm:hidden text-sm font-medium">
            {currentLanguage?.flag}
          </span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {languages.map((lang) => (
          <DropdownMenuItem
            key={lang.code}
            onClick={() => i18n.changeLanguage(lang.code)}
            className="cursor-pointer"
            aria-label={`Switch to ${lang.name}`}
          >
            <span className="mr-2">{lang.flag}</span>
            <span>{lang.name}</span>
            {lang.code === i18n.language && (
              <span className="ml-auto text-xs text-muted-foreground">
                ✓
              </span>
            )}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export default LanguageSelector;
