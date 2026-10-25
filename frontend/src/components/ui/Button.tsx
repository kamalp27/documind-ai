import React from "react";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "danger" | "ghost";
  size?: "sm" | "md" | "lg";
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = "primary",
  size = "md",
  isLoading = false,
  leftIcon,
  rightIcon,
  className = "",
  disabled,
  ...props
}) => {
  const baseStyle: React.CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "0.5rem",
    fontWeight: 600,
    borderRadius: "var(--radius-md)",
    transition: "all var(--transition-fast)",
    cursor: disabled || isLoading ? "not-allowed" : "pointer",
    opacity: disabled || isLoading ? 0.6 : 1,
    border: "1px solid transparent",
  };

  const sizeStyles: Record<string, React.CSSProperties> = {
    sm: { padding: "0.4rem 0.75rem", fontSize: "0.8125rem" },
    md: { padding: "0.6rem 1.1rem", fontSize: "0.875rem" },
    lg: { padding: "0.75rem 1.5rem", fontSize: "1rem" },
  };

  const variantStyles: Record<string, React.CSSProperties> = {
    primary: {
      background: "var(--accent-gradient)",
      color: "var(--text-inverse)",
      boxShadow: "var(--shadow-sm)",
    },
    secondary: {
      backgroundColor: "var(--bg-surface)",
      borderColor: "var(--border-subtle)",
      color: "var(--text-primary)",
    },
    danger: {
      backgroundColor: "var(--status-error-bg)",
      borderColor: "rgba(244, 63, 94, 0.3)",
      color: "var(--status-error)",
    },
    ghost: {
      backgroundColor: "transparent",
      color: "var(--text-secondary)",
    },
  };

  return (
    <button
      style={{ ...baseStyle, ...sizeStyles[size], ...variantStyles[variant] }}
      disabled={disabled || isLoading}
      className={className}
      {...props}
    >
      {isLoading ? (
        <span
          style={{
            display: "inline-block",
            width: "1rem",
            height: "1rem",
            border: "2px solid currentColor",
            borderRightColor: "transparent",
            borderRadius: "50%",
            animation: "spin 0.6s linear infinite",
          }}
        />
      ) : (
        leftIcon
      )}
      {children}
      {!isLoading && rightIcon}
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </button>
  );
};
