export interface NavItem {
  id: string;
  label: string;
  path: string;
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
      { id: "overview", label: "Overview", path: "/overview" },
      { id: "arena", label: "Arena", path: "/arena" },
      { id: "environment-lab", label: "Environment Lab", path: "/environment-lab" },
      { id: "replay", label: "Replay Lab", path: "/replay" },
    ],
  },
  {
    id: "research",
    label: "RESEARCH",
    items: [
      { id: "qualification", label: "Qualification", path: "/qualification" },
      { id: "training", label: "Training", path: "/training" },
      { id: "experiments", label: "Experiments", path: "/experiments" },
      { id: "models", label: "Models", path: "/models" },
      { id: "population", label: "Population", path: "/population" },
      { id: "explainability", label: "Explainability", path: "/explainability" },
    ],
  },
  {
    id: "deliver",
    label: "DELIVER",
    items: [
      { id: "champion", label: "Champion", path: "/champion" },
      { id: "submission", label: "Submission", path: "/submission" },
      { id: "competition", label: "Competition", path: "/competition" },
    ],
  },
  {
    id: "system",
    label: "SYSTEM",
    items: [
      { id: "repository", label: "Repository", path: "/repository" },
      { id: "documentation", label: "Documentation", path: "/documentation" },
    ],
  },
];

export const allNavItems = navGroups.flatMap((g) => g.items);
