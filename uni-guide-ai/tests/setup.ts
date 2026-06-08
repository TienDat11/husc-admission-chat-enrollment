import '@testing-library/jest-dom';

// jsdom doesn't implement matchMedia; ThemeProvider/useTheme reads it at init.
// Provide a minimal stub so the hook returns 'light' by default.
if (typeof window !== 'undefined' && typeof window.matchMedia !== 'function') {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });
}

// IntersectionObserver stub (used by some framer-motion in-view hooks)
if (typeof window !== 'undefined' && typeof (window as any).IntersectionObserver !== 'function') {
  (window as any).IntersectionObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
    takeRecords() { return []; }
  };
}

// jsdom does not implement scrollIntoView on Element prototype. ChatMessages
// calls it from a useEffect. Stub it.
if (typeof Element !== 'undefined' && !(Element.prototype as any).scrollIntoView) {
  (Element.prototype as any).scrollIntoView = function () {};
}
