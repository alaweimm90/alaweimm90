import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/ui/molecules/Tabs";
import { MobileStandardsVerifier } from '@/components/mobile/MobileStandardsVerifier';
import { AccessibilityComplianceChecker } from '@/components/accessibility/AccessibilityComplianceChecker';
import { Smartphone, Users } from 'lucide-react';

export function MobileAccessibilityDashboard() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Mobile & Accessibility Standards</h1>
        <p className="text-muted-foreground">
          WCAG 2.1 AA Compliance • Apple HIG • Material Design • ADA Section 508
        </p>
      </div>

      <Tabs defaultValue="mobile" className="space-y-6">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="mobile" className="flex items-center gap-2">
            <Smartphone className="h-4 w-4" />
            Mobile Standards
          </TabsTrigger>
          <TabsTrigger value="accessibility" className="flex items-center gap-2">
            <Users className="h-4 w-4" />
            Accessibility
          </TabsTrigger>
        </TabsList>

        <TabsContent value="mobile">
          <MobileStandardsVerifier />
        </TabsContent>

        <TabsContent value="accessibility">
          <AccessibilityComplianceChecker />
        </TabsContent>
      </Tabs>
    </div>
  );
}