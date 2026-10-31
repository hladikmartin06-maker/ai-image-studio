"use client";

import { Check, ChevronDown, Info, TriangleAlert, X } from "lucide-react";
import {
  createContext,
  type ButtonHTMLAttributes,
  type CSSProperties,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";

import { useT } from "@/lib/i18n";

function cx(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(" ");
}
export { cx };

// --- Button ------------------------------------------------------------------------------

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "primary" | "ghost" | "danger";
  size?: "sm" | "md" | "lg";
  loading?: boolean;
  icon?: ReactNode;
  block?: boolean;
};

export function Button({
  variant = "default",
  size = "md",
  loading,
  icon,
  block,
  className,
  children,
  disabled,
  type = "button",
  ...rest
}: ButtonProps) {
  return (
    <button
      type={type}
      className={cx(
        "btn",
        variant !== "default" && `btn-${variant}`,
        size !== "md" && `btn-${size}`,
        block && "btn-block",
        className,
      )}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...rest}
    >
      {loading ? <span className="spinner" aria-hidden /> : icon}
      {children}
    </button>
  );
}

// --- Tooltip -----------------------------------------------------------------------------

export function Tooltip({
  label,
  children,
  side = "top",
}: {
  label: ReactNode;
  children: ReactNode;
  side?: "top" | "bottom" | "right";
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const timer = useRef<number | undefined>(undefined);

  const show = () => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      const el = ref.current?.firstElementChild ?? ref.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      if (side === "right") setPos({ x: r.right + 8, y: r.top + r.height / 2 });
      else if (side === "bottom") setPos({ x: r.left + r.width / 2, y: r.bottom + 8 });
      else setPos({ x: r.left + r.width / 2, y: r.top - 8 });
    }, 350);
  };
  const hide = () => {
    window.clearTimeout(timer.current);
    setPos(null);
  };
  useEffect(() => () => window.clearTimeout(timer.current), []);

  const style: CSSProperties | undefined = pos
    ? side === "right"
      ? { left: pos.x, top: pos.y, transform: "translateY(-50%)" }
      : side === "bottom"
        ? { left: pos.x, top: pos.y, transform: "translateX(-50%)" }
        : { left: pos.x, top: pos.y, transform: "translate(-50%, -100%)" }
    : undefined;

  return (
    <span
      ref={ref}
      style={{ display: "contents" }}
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={(e) => {
        // Only keyboard focus shows the tooltip (not focus moved by a dialog or a click).
        if ((e.target as HTMLElement).matches?.(":focus-visible")) show();
      }}
      onBlur={hide}
      onMouseDown={hide}
    >
      {children}
      {pos && label
        ? createPortal(
            <div className="tooltip" role="tooltip" style={style}>
              {label}
            </div>,
            document.body,
          )
        : null}
    </span>
  );
}

// --- IconButton --------------------------------------------------------------------------

export function IconButton({
  label,
  children,
  className,
  onMedia,
  tooltipSide,
  type = "button",
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  label: string;
  onMedia?: boolean;
  tooltipSide?: "top" | "bottom" | "right";
}) {
  return (
    <Tooltip label={label} side={tooltipSide}>
      <button
        type={type}
        aria-label={label}
        className={cx("icon-btn", onMedia && "on-media", className)}
        {...rest}
      >
        {children}
      </button>
    </Tooltip>
  );
}

// --- Switch, Segmented, Chips --------------------------------------------------------------

export function Switch({
  checked,
  onChange,
  disabled,
  label,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  disabled?: boolean;
  label: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      className="switch"
      disabled={disabled}
      onClick={() => onChange(!checked)}
    />
  );
}

export interface Option<T extends string | number> {
  value: T;
  label: ReactNode;
  disabled?: boolean;
  hint?: string;
}

export function Segmented<T extends string | number>({
  value,
  options,
  onChange,
  block,
  label,
}: {
  value: T | null | undefined;
  options: Option<T>[];
  onChange: (value: T) => void;
  block?: boolean;
  label: string;
}) {
  return (
    <div className={cx("segmented", block && "block")} role="group" aria-label={label}>
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          aria-pressed={o.value === value}
          disabled={o.disabled}
          title={o.hint}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

// --- Badge, Spinner, Skeleton, Empty, Notice ---------------------------------------------------

export function Badge({
  tone,
  children,
  className,
  title,
}: {
  tone?: "accent" | "success" | "warning" | "danger" | "outline";
  children: ReactNode;
  className?: string;
  title?: string;
}) {
  return (
    <span className={cx("badge", tone, className)} title={title}>
      {children}
    </span>
  );
}

export function Spinner({ size = 16 }: { size?: number }) {
  return <span className="spinner" style={{ width: size, height: size }} aria-hidden />;
}

export function Skeleton({ style, className }: { style?: CSSProperties; className?: string }) {
  return <div className={cx("skeleton", className)} style={style} aria-hidden />;
}

export function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon: ReactNode;
  title: string;
  description?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <div className="empty-icon">{icon}</div>
      <div className="h2">{title}</div>
      {description ? <p>{description}</p> : null}
      {action ? <div style={{ marginTop: 8 }}>{action}</div> : null}
    </div>
  );
}

export function Notice({
  tone,
  children,
  icon,
}: {
  tone?: "warning" | "danger" | "accent";
  children: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <div className={cx("notice", tone)} role={tone === "danger" ? "alert" : undefined}>
      {icon ?? (tone === "danger" || tone === "warning" ? <TriangleAlert /> : <Info />)}
      <div className="grow">{children}</div>
    </div>
  );
}

export function Field({
  label,
  hint,
  error,
  children,
  htmlFor,
  aside,
}: {
  label: ReactNode;
  hint?: ReactNode;
  error?: string | null;
  children: ReactNode;
  htmlFor?: string;
  aside?: ReactNode;
}) {
  return (
    <div className="field">
      <div className="row">
        <label className="label grow" htmlFor={htmlFor}>
          {label}
        </label>
        {aside}
      </div>
      {children}
      {error ? <div className="error-text">{error}</div> : hint ? <div className="hint">{hint}</div> : null}
    </div>
  );
}

// --- Modal ---------------------------------------------------------------------------------

export function Modal({
  open,
  onClose,
  title,
  description,
  children,
  footer,
  wide,
  icon,
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  description?: ReactNode;
  children?: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
  icon?: ReactNode;
}) {
  const t = useT();
  const panel = useRef<HTMLDivElement>(null);
  // The latest onClose without re-running the focus effect (callers pass inline functions,
  // and a re-render of the page must not move the focus back to the first field).
  const closeRef = useRef(onClose);
  useEffect(() => {
    closeRef.current = onClose;
  });
  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        closeRef.current();
      }
    };
    window.addEventListener("keydown", onKey);
    // First a field, then the main action; the close button only as a last resort.
    const selectors = ["input, textarea", "button.btn-primary, button.btn-danger", "button"];
    for (const selector of selectors) {
      const el = panel.current?.querySelector<HTMLElement>(selector);
      if (el) {
        el.focus();
        break;
      }
    }
    return () => {
      window.removeEventListener("keydown", onKey);
      previous?.focus?.();
    };
  }, [open]);
  if (!open) return null;
  return createPortal(
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div
        ref={panel}
        className={cx("modal", wide && "wide")}
        role="dialog"
        aria-modal="true"
        aria-label={typeof title === "string" ? title : undefined}
      >
        <div className="modal-head">
          {icon}
          <div className="grow">
            <div className="h2">{title}</div>
            {description ? (
              <div className="muted small" style={{ marginTop: 4 }}>
                {description}
              </div>
            ) : null}
          </div>
          <IconButton label={t("common.close")} onClick={onClose}>
            <X />
          </IconButton>
        </div>
        {children ? <div className="modal-body">{children}</div> : null}
        {footer ? <div className="modal-foot">{footer}</div> : null}
      </div>
    </div>,
    document.body,
  );
}

// --- Confirm dialog (promise based) -----------------------------------------------------------

interface ConfirmOptions {
  title: string;
  description?: ReactNode;
  confirmLabel?: string;
  danger?: boolean;
}

const ConfirmContext = createContext<(o: ConfirmOptions) => Promise<boolean>>(async () => false);

export function ConfirmProvider({ children }: { children: ReactNode }) {
  const t = useT();
  const [state, setState] = useState<(ConfirmOptions & { resolve: (v: boolean) => void }) | null>(
    null,
  );
  const confirm = useCallback(
    (options: ConfirmOptions) => new Promise<boolean>((resolve) => setState({ ...options, resolve })),
    [],
  );
  const close = (value: boolean) => {
    state?.resolve(value);
    setState(null);
  };
  return (
    <ConfirmContext.Provider value={confirm}>
      {children}
      <Modal
        open={state !== null}
        onClose={() => close(false)}
        title={state?.title ?? ""}
        description={state?.description}
        footer={
          <>
            <Button variant="ghost" onClick={() => close(false)}>
              {t("common.cancel")}
            </Button>
            <Button variant={state?.danger ? "danger" : "primary"} onClick={() => close(true)}>
              {state?.confirmLabel ?? t("common.confirm")}
            </Button>
          </>
        }
      />
    </ConfirmContext.Provider>
  );
}

export function useConfirm() {
  return useContext(ConfirmContext);
}

// --- Popover / Menu / Select -------------------------------------------------------------------

export function Popover({
  trigger,
  children,
  align = "start",
  width,
}: {
  trigger: (props: { open: boolean; toggle: () => void; ref: (el: HTMLElement | null) => void }) => ReactNode;
  children: (close: () => void) => ReactNode;
  align?: "start" | "end";
  width?: number;
}) {
  const [open, setOpen] = useState(false);
  // State, not a ref: the trigger is rendered by a render prop and receives the setter.
  const [anchor, setAnchor] = useState<HTMLElement | null>(null);
  const panel = useRef<HTMLDivElement>(null);
  const [style, setStyle] = useState<CSSProperties>({});

  const place = useCallback(() => {
    const el = anchor;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const w = width ?? Math.max(r.width, 200);
    let left = align === "end" ? r.right - w : r.left;
    left = Math.max(8, Math.min(left, window.innerWidth - w - 8));
    const below = window.innerHeight - r.bottom;
    const top = below < 260 && r.top > below ? undefined : r.bottom + 6;
    const bottom = top === undefined ? window.innerHeight - r.top + 6 : undefined;
    setStyle({ left, top, bottom, width: w });
  }, [align, width, anchor]);

  useLayoutEffect(() => {
    // Measuring the trigger and positioning the panel before paint is what layout effects are for.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (open) place();
  }, [open, place]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (panel.current?.contains(target) || anchor?.contains(target)) return;
      setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    const onScroll = (e: Event) => {
      if (panel.current?.contains(e.target as Node)) return;
      place();
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    window.addEventListener("resize", place);
    window.addEventListener("scroll", onScroll, true);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", onScroll, true);
    };
  }, [open, place, anchor]);

  const close = useCallback(() => setOpen(false), []);
  const toggle = useCallback(() => setOpen((v) => !v), []);
  return (
    <>
      {trigger({ open, toggle, ref: setAnchor })}
      {open
        ? createPortal(
            <div ref={panel} className="popover" style={style} role="menu">
              {children(close)}
            </div>,
            document.body,
          )
        : null}
    </>
  );
}

export function MenuItem({
  icon,
  children,
  onClick,
  danger,
  active,
  disabled,
  check,
}: {
  icon?: ReactNode;
  children: ReactNode;
  onClick?: () => void;
  danger?: boolean;
  active?: boolean;
  disabled?: boolean;
  check?: boolean;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      className={cx("menu-item", danger && "danger")}
      data-active={active || undefined}
      disabled={disabled}
      onClick={onClick}
    >
      {icon}
      <span className="grow">{children}</span>
      {check ? <Check className="check" /> : null}
    </button>
  );
}

export function Select<T extends string | number>({
  value,
  options,
  onChange,
  label,
  icon,
  placeholder,
  width,
  highlight,
}: {
  value: T | null | undefined;
  options: Option<T>[];
  onChange: (value: T) => void;
  label: string;
  icon?: ReactNode;
  placeholder?: string;
  width?: number;
  highlight?: boolean;
}) {
  const current = options.find((o) => o.value === value);
  return (
    <Popover
      width={width}
      trigger={({ toggle, ref, open }) => (
        <button
          ref={ref}
          type="button"
          className={cx("select-trigger", highlight && "active")}
          aria-haspopup="menu"
          aria-expanded={open}
          aria-label={label}
          onClick={toggle}
          style={width ? { width } : undefined}
        >
          {icon}
          <span className="value truncate">{current?.label ?? placeholder ?? label}</span>
          <ChevronDown />
        </button>
      )}
    >
      {(close) =>
        options.map((o) => (
          <MenuItem
            key={String(o.value)}
            disabled={o.disabled}
            check={o.value === value}
            onClick={() => {
              onChange(o.value);
              close();
            }}
          >
            <div>{o.label}</div>
            {o.hint ? <div className="tiny subtle">{o.hint}</div> : null}
          </MenuItem>
        ))
      }
    </Popover>
  );
}

export function MultiSelect<T extends string>({
  values,
  options,
  onChange,
  label,
  icon,
  allLabel,
}: {
  values: T[];
  options: Option<T>[];
  onChange: (values: T[]) => void;
  label: string;
  icon?: ReactNode;
  allLabel: string;
}) {
  const t = useT();
  const summary =
    values.length === 0
      ? allLabel
      : values.length === 1
        ? options.find((o) => o.value === values[0])?.label
        : t("common.nSelected", { n: values.length });
  return (
    <Popover
      trigger={({ toggle, ref, open }) => (
        <button
          ref={ref}
          type="button"
          className={cx("select-trigger", values.length > 0 && "active")}
          aria-haspopup="menu"
          aria-expanded={open}
          aria-label={label}
          onClick={toggle}
        >
          {icon}
          <span className="value truncate">
            <span className="subtle">{label}: </span>
            {summary}
          </span>
          <ChevronDown />
        </button>
      )}
    >
      {() => (
        <>
          {options.map((o) => {
            const on = values.includes(o.value);
            return (
              <MenuItem
                key={o.value}
                check={on}
                onClick={() => onChange(on ? values.filter((v) => v !== o.value) : [...values, o.value])}
              >
                {o.label}
              </MenuItem>
            );
          })}
          {values.length > 0 ? (
            <>
              <div className="menu-sep" />
              <MenuItem onClick={() => onChange([])}>{t("common.clearSelection")}</MenuItem>
            </>
          ) : null}
        </>
      )}
    </Popover>
  );
}
