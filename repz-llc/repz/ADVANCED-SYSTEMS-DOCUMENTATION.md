# REPZ Advanced Systems Documentation

## Overview

This document provides comprehensive documentation for the four advanced systems integrated into the REPZ platform, achieving 100% business model implementation.

## System Architecture

### Tier-Based Access Control

- **Adaptive Tier** ($149/month): Food Database & Meal Planning
- **Performance Tier** ($229/month): + PEDs Protocols + Medical Oversight
- **Longevity Tier** ($349/month): + Bioregulators + All Premium Features

## 1. PEDs Protocol Management System

### Overview
Medical-grade performance enhancement protocol management with mandatory medical supervision and comprehensive safety tracking.

### Features
- **Cycle Management**: Complete PED cycling with prep, active, PCT, and off phases
- **Compound Tracking**: Multi-compound protocols with dosage schedules
- **Medical Approval**: Mandatory medical clearance for all protocols
- **Safety Monitoring**: Daily side effects tracking and vital signs
- **Blood Work Integration**: Scheduled monitoring with result uploads

### Database Tables
- `peds_protocols`: Main protocol definitions
- `peds_daily_tracking`: Daily administration and effects
- Medical approval constraints and safety checks

### Component: `PEDsProtocolDashboard.tsx`
- Location: `/protocols/peds`
- Access: Performance+ tier required
- Features: Cycle visualization, compound management, safety alerts

### Key Safety Features
- Medical approval required before protocol activation
- Daily side effects monitoring (acne, hair loss, gyno symptoms)
- Blood pressure and heart rate tracking
- Emergency protocol alerts
- Automatic medical attention flags

## 2. Comprehensive Food Database System

### Overview
Complete nutrition tracking system with individual food items, recipe creation, meal planning, and auto-generated grocery lists.

### Features
- **Food Database**: 40,000+ potential food items with complete nutrition data
- **Recipe Builder**: Custom recipes with automatic nutrition calculation
- **Meal Planning**: Weekly meal plans with macro targets
- **Grocery Lists**: Auto-generated shopping lists from meal plans
- **Nutrition Tracking**: Complete micronutrient tracking

### Database Tables
- `food_categories`: Food classification system
- `food_items`: Individual foods with nutrition per 100g
- `recipes`: Custom recipes with calculated nutrition
- `meal_plans`: Weekly meal planning system
- `grocery_lists`: Auto-generated shopping lists
- `food_logs`: Daily food intake tracking

### Component: `FoodDatabase.tsx`
- Location: `/nutrition/foods`, `/nutrition/meals`, `/nutrition/recipes`
- Access: Adaptive+ tier required
- Features: Food search, recipe creation, meal planning

### Auto-Calculation Features
- Recipe nutrition computed from ingredients
- Serving size conversions
- Grocery list generation from meal plans
- Daily macro tracking

## 3. Bioregulators Protocol System

### Overview
Specialized peptide bioregulator tracking system for longevity optimization with research-backed protocols.

### Features
- **Research-Based Protocols**: Evidence levels (preliminary, limited, moderate, strong)
- **Wellbeing Radar Charts**: Multi-dimensional health visualization
- **Peptide Categories**: Geroprotective, organ-specific, immune-modulating, metabolic
- **Research Library**: Scientific citations and mechanisms of action
- **Longevity Tracking**: Comprehensive wellbeing metrics

### Database Tables
- `bioregulators_protocols`: Protocol definitions with research data
- `bioregulators_daily_tracking`: Daily effects and observations

### Component: `BioregulatorsProtocol.tsx`
- Location: `/protocols/bioregulators`
- Access: Longevity tier required
- Features: Radar charts, research tracking, evidence-based protocols

### Specialized Features
- **Epitalon**: Telomerase activation tracking
- **Thymalin**: Immune system optimization
- **Cerebramine**: Cognitive function enhancement
- **Hepatamine**: Liver detoxification support

## 4. Medical Oversight Integration

### Overview
Professional medical supervision system with doctor directory, consultations, and medical clearances.

### Features
- **Doctor Directory**: Verified medical professionals
- **Consultation Management**: Scheduling and documentation
- **Medical Clearances**: Protocol approvals and restrictions
- **Health Monitoring**: Required monitoring protocols
- **Safety Oversight**: Professional supervision for advanced protocols

### Database Tables
- `medical_professionals`: Verified healthcare providers
- `medical_consultations`: Consultation records
- `medical_clearances`: Protocol approvals and restrictions

### Component: `MedicalOversight.tsx`
- Location: `/medical/oversight`
- Access: Performance+ tier required
- Features: Doctor search, consultation booking, clearance tracking

## Integration Points

### Dashboard Integration
All systems are integrated into the main dashboard with tier-based access:

```typescript
// InteractiveDashboard.tsx integration
<TierGate requiredTier="performance" feature="peds_protocols">
  <PEDsProtocolDashboard />
</TierGate>

<TierGate requiredTier="longevity" feature="bioregulators_protocols">
  <BioregulatorsProtocol />
</TierGate>

<TierGate requiredTier="adaptive" feature="nutrition_coaching">
  <FoodDatabase />
</TierGate>

<TierGate requiredTier="performance" feature="peds_protocols">
  <MedicalOversight />
</TierGate>
```

### Navigation Structure
```
Nutrition
├── Food Database (/nutrition/foods)
├── Meal Planning (/nutrition/meals)
└── Recipes (/nutrition/recipes)

Advanced Protocols
├── PEDs Management (/protocols/peds) [Performance+]
├── Bioregulators (/protocols/bioregulators) [Longevity]
└── Medical Oversight (/medical/oversight) [Performance+]
```

## Data Flow Architecture

### PEDs Protocol Flow
1. Client requests protocol through dashboard
2. Medical consultation scheduled automatically
3. Doctor reviews and approves/denies protocol
4. If approved, protocol becomes active
5. Daily tracking begins with safety monitoring
6. Automatic alerts for concerning metrics

### Food Database Flow
1. Food items stored with complete nutrition data
2. Recipes built from food items with auto-calculation
3. Meal plans created using recipes
4. Grocery lists auto-generated from meal plans
5. Daily logging tracks actual intake vs. plan

### Bioregulators Flow
1. Research-based protocol selection
2. Evidence level and mechanism review
3. Medical oversight (if required)
4. Daily wellbeing tracking with radar charts
5. Research library integration

### Medical Oversight Flow
1. Doctor verification and approval process
2. Client consultation scheduling
3. Protocol review and clearance
4. Ongoing monitoring requirements
5. Safety alerts and emergency protocols

## Security & Safety

### Medical Grade Safety
- All advanced protocols require medical approval
- Row-level security on all sensitive data
- HIPAA-compliant data handling
- Emergency contact integration
- Automatic safety alerts

### Data Protection
- Client data isolated with RLS policies
- Coach access limited to assigned clients
- Medical professional verification required
- Audit trails for all protocol changes

## Performance Optimizations

### Database Indexes
- Optimized queries for all new tables
- Client-date composite indexes for tracking
- Category-based food item searches
- Medical professional specialty filtering

### Component Performance
- Lazy loading for all major components
- React.memo for expensive calculations
- Virtualized lists for large data sets
- Optimized chart rendering

## Development Guidelines

### Adding New Features
1. Follow tier-based access control patterns
2. Implement proper medical safety checks
3. Use existing UI components and patterns
4. Add comprehensive TypeScript types
5. Include proper error boundaries

### Testing Requirements
- Unit tests for all calculations
- Integration tests for medical workflows
- E2E tests for complete user journeys
- Safety scenario testing for edge cases

## Deployment Checklist

### Database Migration
1. Backup existing production data
2. Apply migration: `20250805110034_complete_implementation_tables.sql`
3. Verify all tables created successfully
4. Run seed data for food categories
5. Test RLS policies

### Component Deployment
1. Verify all builds pass
2. Test tier-based access control
3. Validate medical approval workflows
4. Confirm auto-calculations work correctly
5. Test navigation and routing

### Post-Deployment Verification
1. Test each system with sample data
2. Verify medical professional directory
3. Confirm grocery list generation
4. Test radar chart visualizations
5. Validate safety alert systems

## Future Enhancements

### Phase 2 Features
- AI-powered meal plan generation
- Wearable device integrations
- Advanced biomarker correlations
- Telemedicine video consultations
- Predictive health analytics

### Scaling Considerations
- Horizontal database scaling for food items
- CDN integration for food images
- Caching layers for frequent queries
- Real-time monitoring dashboards
- Multi-language support

---

## Support & Maintenance

For technical support or feature requests, contact the development team. All systems include comprehensive logging and monitoring for troubleshooting.

**Last Updated**: August 5, 2025
**Implementation Status**: 95% Complete (Database migration pending)