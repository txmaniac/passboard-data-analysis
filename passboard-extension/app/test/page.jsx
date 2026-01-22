"use client";
import React, { useState, useEffect } from 'react';
import PasswordOverlay from '../../components/PasswordOverlay';

export default function TestPage() {
    const [activeElement, setActiveElement] = useState(null);
    const [blockNumbers, setBlockNumbers] = useState(false); // Legacy toggle
    const [useRiskEngine, setUseRiskEngine] = useState(true); // New toggle
    const [blockedKeys, setBlockedKeys] = useState([]);

    // Import the API
    const { fetchBlockedKeys } = require('../../services/riskApi');

    useEffect(() => {
        const handleFocus = (e) => {
            if (e.target.tagName === 'INPUT' && e.target.type === 'password') {
                setActiveElement(e.target);
            }
        };

        const handleMouseDown = (e) => {
            if (!activeElement) return;
            if (e.target === activeElement) return;
            const isOverlay = e.target.closest('.passboard-overlay-check');
            if (isOverlay) return;
            setActiveElement(null);
            setBlockedKeys([]); // Reset on blur
        };

        document.addEventListener('focus', handleFocus, true);
        document.addEventListener('mousedown', handleMouseDown, true);

        return () => {
            document.removeEventListener('focus', handleFocus, true);
            document.removeEventListener('mousedown', handleMouseDown, true);
        };
    }, [activeElement]);

    // Risk Engine Logic
    useEffect(() => {
        if (!activeElement || !useRiskEngine) {
            setBlockedKeys(blockNumbers ? ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'] : []);
            return;
        }

        let debounceTimer;

        const updateRisk = async () => {
            const val = activeElement.value;
            if (val.length === 0) {
                setBlockedKeys([]);
                return;
            }

            // Call API
            const keys = await fetchBlockedKeys(val, true); // Adaptive=True
            setBlockedKeys(keys);
        };

        const handleInput = () => {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(updateRisk, 50); // 50ms Debounce
        };

        // Initial check on focus
        updateRisk();

        activeElement.addEventListener('input', handleInput);
        // Also listen for selection/click as it might change cursor context (though our API is prefix based mainly)

        return () => {
            activeElement.removeEventListener('input', handleInput);
            clearTimeout(debounceTimer);
        };
    }, [activeElement, useRiskEngine, blockNumbers]);

    return (
        <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
            <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-md">
                <h1 className="text-2xl font-bold mb-6 text-gray-800">Signup Test</h1>

                {/* Test Controls */}
                <div className="mb-4 p-3 bg-gray-50 border border-gray-200 rounded space-y-2">
                    <label className="flex items-center gap-2 cursor-pointer text-gray-700">
                        <input
                            type="checkbox"
                            checked={useRiskEngine}
                            onChange={(e) => setUseRiskEngine(e.target.checked)}
                            className="w-4 h-4"
                        />
                        <span className="font-semibold text-blue-600">Enable Smart Risk Blocking</span>
                    </label>

                    <label className="flex items-center gap-2 cursor-pointer text-gray-700 text-sm ml-6 opacity-75">
                        (Legacy) Block Numbers
                        <input
                            type="checkbox"
                            checked={blockNumbers}
                            onChange={(e) => setBlockNumbers(e.target.checked)}
                            className="w-4 h-4 ml-2"
                            disabled={useRiskEngine}
                        />
                    </label>
                </div>

                <div className="space-y-4">
                    {/* ... existing form inputs ... */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
                        <input
                            type="text"
                            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-black"
                            placeholder="johndoe"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
                        <input
                            type="password"
                            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-black"
                            placeholder="••••••••"
                        />
                        <p className="text-xs text-gray-500 mt-1">Focus here to see the overlay.</p>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Confirm Password</label>
                        <input
                            type="password"
                            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-black"
                            placeholder="••••••••"
                        />
                    </div>

                    <button className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition-colors">
                        Sign Up
                    </button>
                </div>
            </div>

            {/* The Overlay Component */}
            <PasswordOverlay targetElement={activeElement} blockedKeys={blockedKeys} />

            <div className="mt-8 p-4 bg-yellow-50 border border-yellow-200 rounded-lg max-w-md text-sm text-yellow-800">
                <p className="font-bold">Debug Info:</p>
                <p>Active Element: {activeElement ? `Input (type=${activeElement.type})` : 'None'}</p>
                <p>Blocked Count: {blockedKeys.length}</p>
                <p className="text-xs mt-1 break-all">{blockedKeys.join(', ')}</p>
            </div>
        </div>
    );
}
