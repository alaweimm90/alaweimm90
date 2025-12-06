import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export const BlogSystem: React.FC = () => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Blog System</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-muted-foreground">
          Blog functionality is being developed. Please check back later for fitness articles and educational content.
        </p>
      </CardContent>
    </Card>
  );
};