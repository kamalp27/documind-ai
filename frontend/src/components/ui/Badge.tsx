import React from "react";

interface BadgeProps {
  children: React.ReactNode;
  variant?: "default" | "accent" | "success" | "warning" | "error";
  size?: "sm" | "md";
  className?: string;
  onClick?: () => void;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = "default",
  size = "md",
  className = "",
  onClick,
}) => {
  const baseStyle: React.CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    gap: "0.25rem",
    fontWeight: 500,
    borderRadius: "var(--radius-full)",
    lineHeight: 1,
    cursor: onClick ? "pointer" : "default",
    border: "1px solid transparent",
  };

  const sizeStyles = {
    sm: { padding: "0.2rem 0.5rem", fontSize: "0.75rem" },
    md: { padding: "0.3rem 0.65rem", fontSize: "0.8125rem" },
  };

  const variantStyles = {
    default: {
      backgroundColor: "var(--bg-surface)",
      borderColor: "var(--border-subtle)",
      color: "var(--text-secondary)",
    },
    accent: {
      backgroundColor: "var(--accent-muted)",
      borderColor: "var(--border-focus)",
      color: "var(--accent-primary)",
    },
    success: {
      backgroundColor: "var(--status-success-bg)",
      borderColor: "rgba(16, 185, 129, 0.3)",
      color: "var(--status-success)",
    },
    warning: {
      backgroundColor: "var(--status-warning-bg)",
      borderColor: "rgba(245, 158, 11, 0.3)",
      color: "var(--status-warning)",
    },
    error: {
      backgroundColor: "var(--status-error-bg)",
      borderColor: "rgba(244, 63, 94, 0.3)",
      color: "var(--status-error)",
    },
  };

  return (
    <span
      style={{ ...baseStyle, ...sizeStyles[size], ...variantStyles[variant] }}
      className={className}
      onClick={onClick}
    >
      {children}
    </span>
  );
};
