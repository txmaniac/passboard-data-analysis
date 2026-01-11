"use client";
import React, { useState } from 'react';

const LAYOUTS = {
    default: [
        ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
        ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
        ['z', 'x', 'c', 'v', 'b', 'n', 'm']
    ],
    shift: [
        ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
        ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
    ],
    symbols: [
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
        ['@', '#', '$', '%', '&', '*', '-', '+', '=', '('],
        [')', '_', '!', '"', "'", ':', ';', '/', '?', '`']
    ]
};

const COLORS = [
    'bg-red-500', 'bg-orange-500', 'bg-yellow-500', 'bg-green-500', 'bg-blue-500', 'bg-indigo-500', 'bg-purple-500', 'bg-pink-500'
];

export default function Keyboard({ onKeyPress, blockedKeys = [] }) {
    const [layout, setLayout] = useState('default'); // default, shift, symbols
    const [capsLock, setCapsLock] = useState(false);
    const [activeKey, setActiveKey] = useState(null);

    const handlePress = (key, action) => {
        // Visual feedback
        const randomColor = COLORS[Math.floor(Math.random() * COLORS.length)];
        setActiveKey({ key, color: randomColor });
        setTimeout(() => setActiveKey(null), 200);

        if (action) {
            if (action === 'shift') {
                if (layout === 'default') setLayout('shift');
                else if (layout === 'shift') setLayout('default');
            }
            else if (action === 'symbols') {
                setLayout(layout === 'symbols' ? 'default' : 'symbols');
            }
            else if (action === 'backspace') {
                onKeyPress({ type: 'backspace' });
            }
            else if (action === 'space') {
                onKeyPress({ type: 'char', value: ' ' });
            }
            return;
        }

        // Normal Character
        onKeyPress({ type: 'char', value: key });

        // Auto-reset shift if not caps locked (simplification: just reset shift after one char)
        // Actually standard mobile keyboard behavior: Shift -> Uppercase 1 char -> Lowercase
        if (layout === 'shift') {
            setLayout('default');
        }
    };

    const renderKey = (key, label = key, width = 'w-8 sm:w-10', action = null) => {
        const isActive = activeKey?.key === key;
        const isBlocked = !action && blockedKeys.includes(key);

        return (
            <button
                key={key}
                disabled={isBlocked}
                className={`
                  ${width} h-10 sm:h-12 rounded-lg font-bold text-white transition-all duration-100 flex items-center justify-center
                  ${isBlocked
                        ? 'bg-gray-800 text-gray-500 cursor-not-allowed opacity-50'
                        : isActive ? activeKey.color : 'bg-gray-700 hover:bg-gray-600 active:scale-95 shadow-md'
                    }
                  text-xs sm:text-base 
                `}
                onMouseDown={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (!isBlocked) handlePress(key, action);
                }}
            >
                {label}
            </button>
        );
    };

    const currentKeys = LAYOUTS[layout] || LAYOUTS.default;

    return (
        <div
            className="flex flex-col gap-2 p-2 sm:p-4 bg-gray-900 rounded-xl shadow-2xl border border-gray-700 w-max touch-none max-w-full"
            onMouseDown={(e) => {
                e.preventDefault();
                // e.stopPropagation(); // Let it bubble, the listener checks target
            }}
        >
            {currentKeys.map((row, rowIndex) => (
                <div key={rowIndex} className="flex justify-center gap-1">
                    {row.map((k) => renderKey(k))}
                </div>
            ))}

            {/* Function Row */}
            <div className="flex justify-center gap-1 mt-1">
                {renderKey('SHIFT', layout === 'shift' ? '⇧•' : '⇧', 'w-12 sm:w-14', 'shift')}
                {renderKey('?123', layout === 'symbols' ? 'ABC' : '?123', 'w-12 sm:w-14', 'symbols')}
                {renderKey('SPACE', '', 'w-24 sm:w-32', 'space')}
                {renderKey('BACKSPACE', '⌫', 'w-12 sm:w-14', 'backspace')}
            </div>
        </div>
    );
}
