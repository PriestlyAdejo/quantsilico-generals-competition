/* Single source for navigation entries and command-palette entries.
   Colour definitions remain in the CSS token layer. */

export interface NavItem {
  id: string;
  label: string;
  path: string;
  icon: string;
  badge?: string;
  disabled?: boolean;
  disabledReason?: string;
}

export interface NavGroup {
  id: string;
  label: string;
  items: NavItem[];
}

export const navGroups: NavGroup[] = [
  {
    id: "operate",
    label: "OPERATE",
    items: [
      { id: "overview",         label: "Overview",         path: "/overview",         icon: "LayoutDashboard" },
      { id: "arena",            label: "Arena",             path: "/arena",             icon: "Swords" },
      { id: "environment-lab",  label: "Environment Lab",  path: "/environment-lab",  icon: "FlaskConical" },
      { id: "replay",           label: "Replay Lab",        path: "/replay",           icon: "PlayCircle" },
    ],
  },
  {
    id: "research",
    label: "RESEARCH",
    items: [
      { id: "qualification",  label: "Qualification",   path: "/qualification",   icon: "ClipboardCheck" },
      { id: "training",       label: "Training",         path: "/training",        icon: "Cpu" },
      { id: "experiments",    label: "Experiments",      path: "/experiments",     icon: "TestTube" },
      { id: "models",         label: "Models",           path: "/models",          icon: "Brain" },
      { id: "population",     label: "Population",       path: "/population",      icon: "Network" },
      { id: "explainability", label: "Explainability",   path: "/explainability",  icon: "Microscope" },
    ],
  },
  {
    id: "deliver",
    label: "DELIVER",
    items: [
      { id: "champion",    label: "Champion",    path: "/champion",    icon: "Trophy" },
      { id: "submission",  label: "Submission",  path: "/submission",  icon: "Upload" },
      { id: "competition", label: "Competition", path: "/competition", icon: "Medal" },
    ],
  },
  {
    id: "system",
    label: "SYSTEM",
    items: [
      { id: "repository",    label: "Repository",    path: "/repository",    icon: "GitBranch" },
      { id: "documentation", label: "Documentation", path: "/documentation", icon: "BookOpen" },
    ],
  },
];

export const allNavItems: NavItem[] = navGroups.flatMap(g => g.items);
