import {neon} from '@neondatabase/serverless';
const sql=neon(process.env.DATABASE_URL);
await sql`CREATE TABLE IF NOT EXISTS replays (dataset TEXT PRIMARY KEY,label TEXT NOT NULL,manifest_url TEXT NOT NULL,manifest JSONB NOT NULL,metrics JSONB NOT NULL DEFAULT '{}'::jsonb,created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`;
await sql`CREATE INDEX IF NOT EXISTS replays_created_at ON replays(created_at DESC)`;
console.log('Cloud replay database schema ready.');
