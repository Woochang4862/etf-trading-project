import { NextResponse } from 'next/server';
import { generatePredictionStatus } from '@/lib/dummy-data';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const status = generatePredictionStatus();
    return NextResponse.json(status);
  } catch (error) {
    console.error('Error fetching prediction status:', error);
    return NextResponse.json(
      { error: 'Failed to fetch prediction status' },
      { status: 500 }
    );
  }
}
