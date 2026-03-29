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

export default function Keyboard({ onKeyPress, blockedKeys = [] }) {
    const [layout, setLayout] = useState('default'); // default, shift, symbols
    const [activeKey, setActiveKey] = useState(null);

    const handlePress = (key, action) => {
        // Simple visual feedback (grey press)
        setActiveKey(key);
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

        if (layout === 'shift') {
            setLayout('default');
        }
    };

    // Calculate Layer Risks
    const isSymbolLayerRisky = LAYOUTS.symbols.flat().some(k => blockedKeys.includes(k));
    // Check both default and shift for alpha risk
    const isAlphaLayerRisky = [...LAYOUTS.default.flat(), ...LAYOUTS.shift.flat()].some(k => blockedKeys.includes(k));

    const renderKey = (key, label = key, width = 'w-8 sm:w-10', action = null) => {
        const isActive = activeKey === key;
        const isBlocked = !action && blockedKeys.includes(key);

        // Determine if this is a layer switch button that needs a risk indicator
        let showRiskIndicator = false;
        if (action === 'symbols') {
            if (layout === 'symbols' && isAlphaLayerRisky) showRiskIndicator = true; // Button says 'ABC', switching to Alpha
            if (layout !== 'symbols' && isSymbolLayerRisky) showRiskIndicator = true; // Button says '?123', switching to Symbols
        }

        return (
            <button
                key={key}
                disabled={isBlocked}
                className={`
                  ${width} h-10 sm:h-12 rounded-lg font-medium transition-all duration-100 flex items-center justify-center relative
                  ${isBlocked
                        ? 'bg-gray-200 text-gray-400 cursor-not-allowed shadow-none' // Blocked: Greyed out text/bg
                        : isActive
                            ? 'bg-gray-300 transform scale-95'  // Active: Slightly darker grey
                            : 'bg-white text-black hover:bg-gray-50 shadow-md border-b-2 border-gray-200' // Default: White, raised
                    }
                  text-sm sm:text-base select-none
                `}
                onMouseDown={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (!isBlocked) handlePress(key, action);
                }}
            >
                {label}
                {showRiskIndicator && (
                    <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-red-500 ring-1 ring-white" />
                )}
            </button>
        );
    };

    const currentKeys = LAYOUTS[layout] || LAYOUTS.default;

    return (
        <div
            className="flex flex-col gap-2 p-2 sm:p-4 bg-gray-100 rounded-xl shadow-xl border border-gray-300 w-max touch-none max-w-full select-none"
            onMouseDown={(e) => e.preventDefault()}
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
