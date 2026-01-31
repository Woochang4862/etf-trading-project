import { NextResponse } from 'next/server';
import { generateTrainingStatus } from '@/lib/dummy-data';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const status = generateTrainingStatus();
    return NextResponse.json(status);
  } catch (error) {
    console.error('Error fetching training status:', error);
    return NextResponse.json(
      { error: 'Failed to fetch training status' },
      { status: 500 }
    );
  }
}
