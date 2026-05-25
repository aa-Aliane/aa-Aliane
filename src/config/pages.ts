import type { PagesConfig } from "../types";

export const PAGES: PagesConfig = {
    home: {
        title: "About Me",
        subtitle: "Research Engineer in NLP & AI",
        isActive: true,
    },
    blog: {
        title: "Blog",
        subtitle: "Insights into AI, NLP, and Engineering.",
        isActive: true,
    },
    publications: {
        title: "Publications",
        subtitle: "Scientific articles in NLP and Text Processing.",
        isActive: true,
    },
    talks: {
        title: "Talks & Presentations",
        subtitle: "",
        isActive: false,
    },
    projects: {
        title: "Projects",
        subtitle: "Engineering and Research implementations.",
        isActive: true,
    },
    teaching: {
        title: "Teaching",
        subtitle: "",
        isActive: false,
    },
    tags: {
        title: "Tags",
        subtitle: "",
        isActive: false,
    },
    cv: {
        title: "Curriculum Vitae",
        subtitle: "Professional and Academic Journey",
        isActive: true,
    },
};
