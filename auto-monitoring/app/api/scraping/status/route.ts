import { NextResponse } from 'next/server';
import { parseScrapingLog } from '@/lib/scraping-log-parser';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export async function GET() {
  try {
    const status = await parseScrapingLog();
    return NextResponse.json(status);
  } catch (error) {
    console.error('Error fetching scraping status:', error);
    return NextResponse.json(
      {
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
        lastRun: null,
        currentSession: null,
        progress: {
          totalSymbols: 101,
          completedSymbols: 0,
          currentSymbol: null,
          currentTimeframe: null,
          percentage: 0,
        },
        statistics: {
          totalDownloads: 0,
          successfulUploads: 0,
          failedDownloads: 0,
          totalRowsUploaded: 0,
        },
        symbols: [],
        errors: [],
      },
      { status: 500 }
    );
  }
}
