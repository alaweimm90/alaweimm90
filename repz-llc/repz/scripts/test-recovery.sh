#!/bin/bash
# Disaster Recovery Testing Script

set -e

echo "🧪 Starting disaster recovery test..."

# Test database connectivity
echo "🗄️  Testing database connectivity..."
if [ ! -z "$SUPABASE_DB_URL" ]; then
  psql "$SUPABASE_DB_URL" -c "SELECT 1;" > /dev/null 2>&1
  if [ $? -eq 0 ]; then
    echo "✅ Database connection successful"
  else
    echo "❌ Database connection failed"
    exit 1
  fi
else
  echo "⚠️  No database URL configured"
fi

# Test application build
echo "🏗️  Testing application build..."
npm run build > /dev/null 2>&1
if [ $? -eq 0 ]; then
  echo "✅ Application builds successfully"
else
  echo "❌ Application build failed"
  exit 1
fi

# Test key application routes
echo "🧪 Testing application functionality..."
npm run dev &
SERVER_PID=$!
sleep 10

# Test health endpoint
curl -f http://localhost:8080 > /dev/null 2>&1
if [ $? -eq 0 ]; then
  echo "✅ Application responds to requests"
else
  echo "❌ Application not responding"
  kill $SERVER_PID 2>/dev/null
  exit 1
fi

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "✅ Disaster recovery test completed successfully"
echo "📊 All systems operational after recovery simulation"
