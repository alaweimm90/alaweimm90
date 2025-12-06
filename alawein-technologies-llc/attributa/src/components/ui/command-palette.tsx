import * as React from "react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Command } from "cmdk";
import { Search, FileSearch, Settings, History, Code, Book, Zap } from "lucide-react";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const commandItems = [
  {
    group: "Navigation",
    items: [
      { id: "workspace", label: "Workspace", path: "/workspace", icon: History },
      { id: "settings", label: "Settings", path: "/settings", icon: Settings },
      { id: "scan", label: "Analyze Document", path: "/scan", icon: FileSearch },
    ]
  },
  {
    group: "Methods",
    items: [
      { id: "gltr", label: "GLTR Analysis", description: "Statistical likelihood detection", icon: Code },
      { id: "detectgpt", label: "DetectGPT", description: "Perturbation-based detection", icon: Zap },
      { id: "watermark", label: "Watermark Detection", description: "Cryptographic verification", icon: Book },
    ]
  },
  {
    group: "Documentation",
    items: [
      { id: "docs", label: "Documentation", description: "API reference and guides", icon: Book },
      { id: "sdk", label: "Python SDK", description: "Integration examples", icon: Code },
    ]
  }
];

export function CommandPalette({ open, onOpenChange }: CommandPaletteProps) {
  const navigate = useNavigate();
  const [search, setSearch] = useState("");

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        onOpenChange(!open);
      }
    };

    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, [open, onOpenChange]);

  const handleSelect = (path?: string) => {
    if (path) {
      navigate(path);
    }
    onOpenChange(false);
    setSearch("");
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="overflow-hidden p-0 shadow-lg border-border/50">
        <Command className="[&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:font-mono [&_[cmdk-group-heading]]:text-xs [&_[cmdk-group-heading]]:text-muted-foreground [&_[cmdk-group]:not([hidden])_~[cmdk-group]]:pt-0 [&_[cmdk-group]]:px-2 [&_[cmdk-input-wrapper]_svg]:h-5 [&_[cmdk-input-wrapper]_svg]:w-5 [&_[cmdk-input]]:h-12 [&_[cmdk-item]]:px-2 [&_[cmdk-item]]:py-3 [&_[cmdk-item]_svg]:h-5 [&_[cmdk-item]_svg]:w-5">
          <div className="flex items-center border-b border-border/50 px-3">
            <Search className="mr-2 h-4 w-4 shrink-0 text-muted-foreground" />
            <Command.Input
              value={search}
              onValueChange={setSearch}
              placeholder="Search methods, docs, navigation..."
              className="flex h-11 w-full rounded-md bg-transparent py-3 text-sm outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:opacity-50"
            />
            <kbd className="pointer-events-none hidden h-5 select-none items-center gap-1 rounded border border-border/50 bg-muted px-1.5 font-mono text-xs font-medium text-muted-foreground opacity-100 sm:flex">
              <span className="text-xs">⌘</span>K
            </kbd>
          </div>
          <Command.List className="max-h-[300px] overflow-y-auto overflow-x-hidden">
            <Command.Empty className="py-6 text-center text-sm text-muted-foreground">
              No results found.
            </Command.Empty>
            {commandItems.map((group) => (
              <Command.Group key={group.group} heading={group.group}>
                {group.items.map((item) => {
                  const Icon = item.icon;
                  return (
                    <Command.Item
                      key={item.id}
                      value={item.label}
                      onSelect={() => handleSelect(item.path)}
                      className={cn(
                        "flex cursor-pointer select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none aria-selected:bg-accent aria-selected:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50"
                      )}
                    >
                      <Icon className="mr-2 h-4 w-4" />
                      <div className="flex flex-col">
                        <span>{item.label}</span>
                        {item.description && (
                          <span className="text-xs text-muted-foreground">
                            {item.description}
                          </span>
                        )}
                      </div>
                    </Command.Item>
                  );
                })}
              </Command.Group>
            ))}
          </Command.List>
        </Command>
      </DialogContent>
    </Dialog>
  );
}