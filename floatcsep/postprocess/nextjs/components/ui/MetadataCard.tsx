
import { safeRender } from '@/lib/utils';

interface MetadataCardProps {
    title: string;
    data: Array<{ label: string; value: any }>;
    className?: string;
}

export default function MetadataCard({ title, data, className = '' }: MetadataCardProps) {
    // Filter out items with null/undefined values
    const validData = data.filter(item => {
        if (item.value === null || item.value === undefined) return false;
        if (Array.isArray(item.value) && item.value.length === 0) return false;
        return true;
    });

    if (validData.length === 0) return null;

    return (
        <div className={`bg-surface p-6 rounded-lg border border-border ${className}`}>
            <h2 className="text-xl font-semibold mb-4">{title}</h2>
            <div className="space-y-2 text-sm">
                {validData.map((item, idx) => (
                    <p key={idx}>
                        <span className="text-gray-400">{item.label}:</span>{' '}
                        {safeRender(item.value)}
                    </p>
                ))}
            </div>
        </div>
    );
}
