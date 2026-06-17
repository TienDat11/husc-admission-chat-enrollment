import '@testing-library/jest-dom';

type MatchMedia = (query: string) => {
  matches: boolean;
  media: string;
  onchange: null;
  addListener: () => void;
  removeListener: () => void;
  addEventListener: () => void;
  removeEventListener: () => void;
  dispatchEvent: () => boolean;
};

type IntersectionObserverStub = {
  new (): {
    observe: () => void;
    unobserve: () => void;
    disconnect: () => void;
    takeRecords: () => [];
  };
};

// jsdom doesn't implement matchMedia; ThemeProvider/useTheme reads it at init.
// Provide a minimal stub so the hook returns 'light' by default.
if (typeof window !== 'undefined' && typeof window.matchMedia !== 'function') {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string): ReturnType<MatchMedia> => ({
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
if (
  typeof window !== 'undefined' &&
  typeof (window as unknown as { IntersectionObserver?: IntersectionObserverStub }).IntersectionObserver !==
    'function'
) {
  (
    window as unknown as { IntersectionObserver: IntersectionObserverStub }
  ).IntersectionObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
    takeRecords() {
      return [];
    }
  };
}

// jsdom does not implement scrollIntoView on Element prototype. ChatMessages
// calls it from a useEffect. Stub it.
if (
  typeof Element !== 'undefined' &&
  typeof (Element.prototype as unknown as { scrollIntoView?: () => void }).scrollIntoView !==
    'function'
) {
  (Element.prototype as unknown as { scrollIntoView: () => void }).scrollIntoView = function () {};
}
