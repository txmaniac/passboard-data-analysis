import React from 'react';
import { createRoot } from 'react-dom/client';
import PasswordOverlay from '../components/PasswordOverlay';

// Heuristic to detect signup pages
function isSignupPage() {
    const url = window.location.href.toLowerCase();
    const title = document.title.toLowerCase();
    const keywords = ['sign up', 'signup', 'register', 'create account', 'join'];

    if (keywords.some(k => url.includes(k) || title.includes(k))) return true;

    // Check for password inputs
    const passwordInputs = document.querySelectorAll('input[type="password"]');
    // If multiple password inputs (confirm password), likely signup
    if (passwordInputs.length >= 2) return true;

    // Also check if there's a "new password" hint
    for (let input of passwordInputs) {
        if (input.name.includes('new') || input.id.includes('new') || input.autocomplete.includes('new-password')) {
            return true;
        }
    }

    return false;
}

function init() {
    console.log('PassBoard Extension: Checking page type...');

    const isSignup = isSignupPage();
    // Allow manual override for testing or if detection is failing
    const forceEnable = false; // Set to true for development/testing

    if (!isSignup && !forceEnable) {
        console.log('PassBoard Extension: Not a signup page.');
        return;
    }

    console.log('PassBoard Extension: Initializing...');

    const passwordInputs = document.querySelectorAll('input[type="password"]');
    if (passwordInputs.length === 0) return;

    // Create a host for Shadow DOM
    const host = document.createElement('div');
    host.id = 'passboard-extension-root';
    host.style.position = 'absolute';
    host.style.top = '0';
    host.style.left = '0';
    host.style.width = '100%';
    host.style.height = '0';
    host.style.pointerEvents = 'none'; // Allow clicking through
    host.style.zIndex = '2147483647';
    document.body.appendChild(host);

    const shadow = host.attachShadow({ mode: 'open' });

    // Inject CSS
    const styleLink = document.createElement('link');
    styleLink.rel = 'stylesheet';
    styleLink.href = chrome.runtime.getURL('extension.css');
    shadow.appendChild(styleLink);

    // Container for React
    const container = document.createElement('div');
    // Need to re-enable pointer events for the keyboard itself
    // But since the host has pointer-events: none, we need to override on children
    // Actually, standard practice: host is 0x0 fixed or just appended where needed.
    // The Overlay component calculates position.
    shadow.appendChild(container);

    const root = createRoot(container);

    // We need to pass the target inputs to the overlay
    // Or render multiple overlays.
    // For now, let's just pick the first one or manage state to track the focused one.

    const App = () => {
        const [focusedInput, setFocusedInput] = React.useState(null);
        const focusedInputRef = React.useRef(null);

        React.useEffect(() => {
            const handleFocus = (e) => {
                if (e.target.type === 'password') {
                    setFocusedInput(e.target);
                    focusedInputRef.current = e.target;
                }
            };

            const handleMouseDown = (e) => {
                // If we have an active input
                if (focusedInputRef.current) {
                    // If clicking the input itself, do nothing
                    if (e.target === focusedInputRef.current) return;

                    // If clicking inside the extension (shadow host checks)
                    const isExtension = e.composedPath().some(el => el.id === 'passboard-extension-root');
                    if (isExtension) return;

                    // Otherwise, close
                    setFocusedInput(null);
                    focusedInputRef.current = null;
                }
            };

            document.addEventListener('focus', handleFocus, true);
            document.addEventListener('mousedown', handleMouseDown, true);

            return () => {
                document.removeEventListener('focus', handleFocus, true);
                document.removeEventListener('mousedown', handleMouseDown, true);
            };
        }, []);

        return <PasswordOverlay targetElement={focusedInput} />;
    };

    root.render(<App />);
}

// init on load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
