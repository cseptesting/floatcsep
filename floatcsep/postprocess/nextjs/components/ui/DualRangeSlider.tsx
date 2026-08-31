'use client';

import { useEffect, useRef, useState, useCallback } from 'react';

interface DualRangeSliderProps {
    min: number;
    max: number;
    step?: number;
    value: [number, number];
    onValueChange: (value: [number, number]) => void;
    className?: string;
}

export default function DualRangeSlider({
    min,
    max,
    step = 1,
    value,
    onValueChange,
    className = '',
}: DualRangeSliderProps) {
    const [isDragging, setIsDragging] = useState<'min' | 'max' | null>(null);
    const sliderRef = useRef<HTMLDivElement>(null);

    // Helper to convert value to percentage
    const getPercent = useCallback((val: number) => {
        return Math.round(((val - min) / (max - min)) * 100);
    }, [min, max]);

    // Handle move events
    const handleMove = useCallback((clientX: number) => {
        if (!isDragging || !sliderRef.current) return;

        const rect = sliderRef.current.getBoundingClientRect();
        const percent = Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100));

        // Convert percent back to value
        const rawValue = min + (percent / 100) * (max - min);

        // Snap to step
        const steppedValue = Math.round(rawValue / step) * step;
        const clampedValue = Math.min(max, Math.max(min, steppedValue)); // Ensure within bounds (floating point fixes later)

        // Fix floating point precision
        const preciseValue = Number(clampedValue.toFixed(4));

        if (isDragging === 'min') {
            const newValue = Math.min(preciseValue, value[1] - step);
            onValueChange([newValue, value[1]]);
        } else {
            const newValue = Math.max(preciseValue, value[0] + step);
            onValueChange([value[0], newValue]);
        }
    }, [isDragging, min, max, step, value, onValueChange]);

    const handleMouseUp = useCallback(() => {
        setIsDragging(null);
    }, []);

    const handleMouseMove = useCallback((e: MouseEvent) => {
        handleMove(e.clientX);
    }, [handleMove]);

    const handleTouchMove = useCallback((e: TouchEvent) => {
        handleMove(e.touches[0].clientX);
    }, [handleMove]);

    useEffect(() => {
        if (isDragging) {
            window.addEventListener('mousemove', handleMouseMove);
            window.addEventListener('mouseup', handleMouseUp);
            window.addEventListener('touchmove', handleTouchMove);
            window.addEventListener('touchend', handleMouseUp);
        } else {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
            window.removeEventListener('touchmove', handleTouchMove);
            window.removeEventListener('touchend', handleMouseUp);
        }
        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
            window.removeEventListener('touchmove', handleTouchMove);
            window.removeEventListener('touchend', handleMouseUp);
        };
    }, [isDragging, handleMouseMove, handleMouseUp, handleTouchMove]);

    const minPercent = getPercent(value[0]);
    const maxPercent = getPercent(value[1]);

    return (
        <div className={`relative w-full h-6 flex items-center select-none touch-none ${className}`}>
            <div
                ref={sliderRef}
                className="relative w-full h-1 bg-border rounded-full"
            >
                {/* Selected Range Track */}
                <div
                    className="absolute h-full bg-primary rounded-full"
                    style={{ left: `${minPercent}%`, width: `${maxPercent - minPercent}%` }}
                />

                {/* Min Thumb */}
                <div
                    className="absolute w-4 h-4 bg-white border-2 border-primary rounded-full shadow cursor-grab active:cursor-grabbing -ml-2 -mt-1.5 hover:scale-110 transition-transform"
                    style={{ left: `${minPercent}%`, zIndex: value[0] > max - (max - min) / 20 ? 5 : 3 }}
                    onMouseDown={() => setIsDragging('min')}
                    onTouchStart={() => setIsDragging('min')}
                />

                {/* Max Thumb */}
                <div
                    className="absolute w-4 h-4 bg-white border-2 border-primary rounded-full shadow cursor-grab active:cursor-grabbing -ml-2 -mt-1.5 hover:scale-110 transition-transform"
                    style={{ left: `${maxPercent}%`, zIndex: 4 }}
                    onMouseDown={() => setIsDragging('max')}
                    onTouchStart={() => setIsDragging('max')}
                />
            </div>
        </div>
    );
}
