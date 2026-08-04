import React, { useState } from "react";
import { useNavigate } from "react-router";
import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "../app/components/ui/command";
import {
  LayoutDashboard, Swords, FlaskConical, PlayCircle, ClipboardCheck, Cpu,
  TestTube, Brain, Network, Microscope, Trophy, Upload, Medal, GitBranch, BookOpen,
} from "lucide-react";
import { navGroups } from "../app/navigation";
import { useKeyboardShortcut } from "../hooks/useKeyboardShortcut";

const ICONS: Record<string, React.ComponentType<{ size?: number }>> = {
  LayoutDashboard, Swords, FlaskConical, PlayCircle, ClipboardCheck, Cpu,
  TestTube, Brain, Network, Microscope, Trophy, Upload, Medal, GitBranch, BookOpen,
};

export default function CommandPalette() {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();

  useKeyboardShortcut(["k"], () => setOpen((o) => !o), []);

  return (
    <CommandDialog open={open} onOpenChange={setOpen}>
      <Command className="bg-[#11161C] border border-[#1E2630]">
        <CommandInput
          placeholder="Search pages..."
          className="border-b border-[#1E2630] text-[#EAF0F6] placeholder:text-[#6F7C89]"
          style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}
        />
        <CommandList>
          <CommandEmpty
            className="text-[#6F7C89] py-6 text-center"
            style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}
          >
            No results found.
          </CommandEmpty>
          {navGroups.map((group) => (
            <CommandGroup
              key={group.id}
              heading={group.label}
              className="text-[#6F7C89]"
            >
              {group.items.map((item) => {
                const Icon = ICONS[item.icon] ?? LayoutDashboard;
                return (
                  <CommandItem
                    key={item.id}
                    value={item.label}
                    onSelect={() => {
                      navigate(item.path);
                      setOpen(false);
                    }}
                    className="flex items-center gap-2 text-[#8593A1] aria-selected:bg-[#161C24] aria-selected:text-[#FFB000] cursor-pointer"
                    style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}
                  >
                    <Icon size={13} />
                    {item.label}
                  </CommandItem>
                );
              })}
            </CommandGroup>
          ))}
        </CommandList>
      </Command>
    </CommandDialog>
  );
}
