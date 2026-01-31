'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrainingStatus as TrainingStatusType } from '@/lib/types';

interface TrainingStatusProps {
  data: TrainingStatusType;
}

export function TrainingStatus({ data }: TrainingStatusProps) {
  const { status, lastTraining, nextScheduled, models } = data;
  const isTraining = status === 'training';

  return (
    <Card className="border-l-4 border-l-amber-500">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <CardTitle className="flex items-center gap-2 text-xl">
              Model Training
              {isTraining && (
                <div className="flex gap-0.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" style={{ animationDelay: '0ms' }} />
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" style={{ animationDelay: '150ms' }} />
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" style={{ animationDelay: '300ms' }} />
                </div>
              )}
            </CardTitle>
            <CardDescription>Monthly model retraining pipeline</CardDescription>
          </div>
          <Badge variant={getTrainingStatusVariant(status)} className="uppercase text-xs font-bold">
            {status}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Schedule Info */}
        <div className="grid grid-cols-2 gap-3">
          <div className="p-3 rounded-lg bg-muted/30 border border-foreground/5">
            <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
              Last Training
            </div>
            <div className="text-sm font-medium">
              {lastTraining ? new Date(lastTraining).toLocaleDateString() : 'Never'}
            </div>
          </div>
          <div className="p-3 rounded-lg bg-amber-500/5 border border-amber-500/20">
            <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
              Next Scheduled
            </div>
            <div className="text-sm font-medium text-amber-700 dark:text-amber-400">
              {new Date(nextScheduled).toLocaleDateString()}
            </div>
          </div>
        </div>

        {/* Models Grid */}
        <div className="space-y-2">
          <h4 className="text-xs font-bold uppercase tracking-wide text-muted-foreground">
            Active Models ({models.length})
          </h4>
          <div className="grid gap-2">
            {models.map((model, i) => (
              <ModelCard key={i} model={model} />
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function ModelCard({ model }: { model: TrainingStatusType['models'][0] }) {
  const statusColors = {
    pending: 'bg-gray-500/10 border-gray-500/20 text-gray-600 dark:text-gray-400',
    training: 'bg-amber-500/10 border-amber-500/20 text-amber-600 dark:text-amber-400',
    trained: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-600 dark:text-emerald-400',
    failed: 'bg-red-500/10 border-red-500/20 text-red-600 dark:text-red-400',
  };

  return (
    <div className={`p-3 rounded-lg border transition-all hover:shadow-md ${statusColors[model.status]}`}>
      <div className="flex items-start justify-between mb-2">
        <div>
          <div className="font-bold text-sm">{model.name}</div>
          <div className="text-xs text-muted-foreground">{model.symbols} symbols</div>
        </div>
        <Badge variant="outline" className="text-[10px] px-2 uppercase font-bold">
          {model.status}
        </Badge>
      </div>
      <div className="flex items-center justify-between">
        <div className="text-xs text-muted-foreground">
          Updated: {new Date(model.lastUpdated).toLocaleDateString()}
        </div>
        {model.status === 'trained' && (
          <div className="text-xs font-bold">
            {(model.accuracy * 100).toFixed(1)}% acc
          </div>
        )}
      </div>
    </div>
  );
}

function getTrainingStatusVariant(status: string): 'default' | 'secondary' | 'destructive' | 'outline' {
  switch (status) {
    case 'training': return 'default';
    case 'completed': return 'secondary';
    default: return 'outline';
  }
}
