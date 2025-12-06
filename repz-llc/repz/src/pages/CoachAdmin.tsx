import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';

export default function CoachAdmin() {
  return (
    <div className="container mx-auto p-6">
      <Card>
        <CardHeader>
          <CardTitle>Coach Admin</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            Coach admin functionality is being developed. Please check back later.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}