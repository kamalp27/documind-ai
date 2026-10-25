import "@testing-library/jest-dom";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

// Auto cleanup after each test
afterEach(() => {
  cleanup();
});

// Mock scrollIntoView for jsdom
window.HTMLElement.prototype.scrollIntoView = function () {};
