"use client";
import React, { useEffect, useState } from 'react';
import Keyboard from './Keyboard';

export default function PasswordOverlay({ targetElement, blockedKeys = [] }) {
    const [style, setStyle] = useState({ display: 'none' });

    useEffect(() => {
        if (!targetElement) {
            setStyle({ display: 'none' });
            return;
        }

        const updatePosition = () => {
            const rect = targetElement.getBoundingClientRect();

            // Use fixed positioning to simplify parent context issues
            const newStyle = {
                position: 'fixed',
                top: `${rect.bottom + 8}px`,
                left: `${rect.left}px`,
                zIndex: 99999,
                display: 'block',
                pointerEvents: 'auto',
            };
            setStyle(newStyle);
        };

        updatePosition();
        // Re-position on scroll/resize could be added here
        window.addEventListener('scroll', updatePosition);
        window.addEventListener('resize', updatePosition);

        return () => {
            window.removeEventListener('scroll', updatePosition);
            window.removeEventListener('resize', updatePosition);
        };
    }, [targetElement]);


    const handleKeyPress = (event) => {
        // ... helper logic ...
        if (!targetElement) return;

        const input = targetElement;
        const currentVal = input.value;
        const start = input.selectionStart || 0; // Guard
        const end = input.selectionEnd || 0; // Guard

        // Helper to dispatch events
        const updateValue = (newValue, newCursorPos) => {
            const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
            valueSetter.call(input, newValue);
            input.value = newValue; // Fallback
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
            input.focus();
            input.setSelectionRange(newCursorPos, newCursorPos);
        };

        if (event.type === 'char') {
            const newVal = currentVal.substring(0, start) + event.value + currentVal.substring(end);
            updateValue(newVal, start + 1);
        } else if (event.type === 'backspace') {
            if (start === end && start > 0) {
                const newVal = currentVal.substring(0, start - 1) + currentVal.substring(end);
                updateValue(newVal, start - 1);
            } else if (start !== end) {
                const newVal = currentVal.substring(0, start) + currentVal.substring(end);
                updateValue(newVal, start);
            }
        }
    };

    if (!targetElement) return null;

    return (
        <div style={style} className="passboard-overlay-check">
            <Keyboard onKeyPress={handleKeyPress} blockedKeys={blockedKeys} />
        </div>
    );
}
