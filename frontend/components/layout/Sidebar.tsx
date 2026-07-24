"use client";

import {
  Bot,
  Home,
  MessagesSquare,
  Puzzle,
  Search,
  Settings,
  Sparkles,
  Upload,
  Users,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { toast } from "sonner";

import { cn } from "@/lib/utils";
import { useSidebar } from "./SidebarContext";

type NavItem = {
  label: string;
  href?: string;
  icon: React.ComponentType<{ size?: number }>;
  soon?: boolean;
};

const NAV: NavItem[] = [
  { label: "Home", href: "/", icon: Home },
  { label: "Assistant", href: "/assistant", icon: MessagesSquare },
  { label: "Search", href: "/search", icon: Search },
  { label: "Notetaker", icon: Bot, soon: true },
  { label: "Uploads", icon: Upload, soon: true },
  { label: "Integrations", icon: Puzzle, soon: true },
  { label: "Team", icon: Users, soon: true },
  { label: "Settings", href: "/settings", icon: Settings },
];

export default function Sidebar() {
  const pathname = usePathname();
  const { mobileOpen, collapsed, closeMobile } = useSidebar();

  return (
    <>
      {/* Mobile backdrop */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={closeMobile}
          aria-hidden
        />
      )}

      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex w-60 flex-col overflow-hidden border-r border-line bg-card transition-[transform,width] duration-300 ease-in-out",
          // mobile: controlled by the drawer state
          mobileOpen ? "translate-x-0" : "-translate-x-full",
          // desktop: always visible, collapses to a narrow icon rail (Fireflies-style)
          "md:translate-x-0",
          collapsed ? "md:w-16" : "md:w-60",
        )}
      >
        <Link
          href="/"
          className={cn(
            "flex items-center gap-2.5 px-5 py-4",
            collapsed && "md:justify-center md:px-0",
          )}
        >
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-brand to-indigo-500 text-white shadow-sm">
            <Sparkles size={17} />
          </span>
          <span
            className={cn(
              "text-[15px] font-semibold tracking-tight text-ink",
              collapsed && "md:hidden",
            )}
          >
            Fireflies
          </span>
        </Link>

        <nav className="flex-1 space-y-1 px-3 py-2">
          {NAV.map((item) => {
            const active =
              item.href &&
              (item.href === "/" ? pathname === "/" : pathname.startsWith(item.href));
            const className = cn(
              "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition",
              active
                ? "bg-brand-soft text-brand"
                : "text-muted hover:bg-panel hover:text-ink",
              collapsed && "md:justify-center md:px-0",
            );
            const inner = (
              <>
                <item.icon size={18} />
                <span className={cn(collapsed && "md:hidden")}>{item.label}</span>
                {item.soon && (
                  <span
                    className={cn(
                      "ml-auto rounded-full bg-panel px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-muted",
                      collapsed && "md:hidden",
                    )}
                  >
                    Soon
                  </span>
                )}
              </>
            );

            return item.href ? (
              <Link
                key={item.label}
                href={item.href}
                title={item.label}
                className={className}
              >
                {inner}
              </Link>
            ) : (
              <button
                key={item.label}
                type="button"
                title={item.label}
                onClick={() => toast(`${item.label} — coming soon`)}
                className={cn(className, "w-full text-left")}
              >
                {inner}
              </button>
            );
          })}
        </nav>

        <div className="border-t border-line p-3">
          <div
            className={cn(
              "flex items-center gap-3 rounded-lg px-2 py-2",
              collapsed && "md:justify-center md:px-0",
            )}
          >
            <span className="grid h-8 w-8 place-items-center rounded-full bg-gradient-to-br from-emerald-400 to-teal-500 text-xs font-semibold text-white">
              YO
            </span>
            <div className={cn("min-w-0", collapsed && "md:hidden")}>
              <p className="truncate text-sm font-medium text-ink">You</p>
              <p className="truncate text-xs text-muted">Free workspace</p>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
