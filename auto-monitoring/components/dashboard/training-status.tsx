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
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <CardTitle className="flex items-center gap-2 text-lg">
              Model Training
              {isTraining && (
                <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
              )}
            </CardTitle>
            <CardDescription className="text-sm">Monthly model retraining pipeline</CardDescription>
          </div>
          <Badge variant={getTrainingStatusVariant(status)} className="uppercase text-xs">
            {status}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Schedule Info */}
        <div className="grid grid-cols-2 gap-3">
          <div className="p-3 rounded-lg bg-muted/50 border">
            <div className="text-xs font-medium text-muted-foreground mb-1">
              Last Training
            </div>
            <div className="text-sm font-medium">
              {lastTraining ? new Date(lastTraining).toLocaleDateString() : 'Never'}
            </div>
          </div>
          <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-200 dark:border-amber-800">
            <div className="text-xs font-medium text-muted-foreground mb-1">
              Next Scheduled
            </div>
            <div className="text-sm font-medium text-amber-700 dark:text-amber-400">
              {new Date(nextScheduled).toLocaleDateString()}
            </div>
          </div>
        </div>

        {/* Models Grid */}
        <div className="space-y-2">
          <h4 className="text-xs font-medium text-muted-foreground uppercase">
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
    pending: 'bg-muted/50 border text-muted-foreground',
    training: 'bg-amber-500/10 border-amber-200 dark:border-amber-800 text-amber-700 dark:text-amber-400',
    trained: 'bg-green-500/10 border-green-200 dark:border-green-800 text-green-700 dark:text-green-400',
    failed: 'bg-red-500/10 border-red-200 dark:border-red-800 text-red-700 dark:text-red-400',
  };

  return (
    <div className={`p-3 rounded-lg border ${statusColors[model.status]}`}>
      <div className="flex items-start justify-between mb-2">
        <div>
          <div className="font-medium text-sm">{model.name}</div>
          <div className="text-xs text-muted-foreground">{model.symbols} symbols</div>
        </div>
        <Badge variant="outline" className="text-[10px] px-2 uppercase">
          {model.status}
        </Badge>
      </div>
      <div className="flex items-center justify-between">
        <div className="text-xs text-muted-foreground">
          Updated: {new Date(model.lastUpdated).toLocaleDateString()}
        </div>
        {model.status === 'trained' && (
          <div className="text-xs font-semibold">
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
