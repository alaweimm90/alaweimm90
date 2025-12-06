import React from 'react';
import { useLocation, Link } from 'react-router-dom';
import {
  Breadcrumb,
  BreadcrumbList,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';

function toTitleCase(segment: string) {
  return segment
    .replace(/-/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export const AutoBreadcrumbs: React.FC<{ className?: string }> = ({ className }) => {
  const location = useLocation();
  const segments = location.pathname.split('/').filter(Boolean);

  const paths = segments.map((seg, i) => ({
    name: toTitleCase(seg),
    href: '/' + segments.slice(0, i + 1).join('/'),
  }));

  return (
    <nav aria-label="Breadcrumb" className={className}>
      <Breadcrumb>
        <BreadcrumbList>
          <BreadcrumbItem>
            <BreadcrumbLink asChild>
              <Link to="/">Home</Link>
            </BreadcrumbLink>
          </BreadcrumbItem>
          {paths.length > 0 && <BreadcrumbSeparator />}
          {paths.map((p, idx) => (
            <React.Fragment key={p.href}>
              <BreadcrumbItem>
                {idx < paths.length - 1 ? (
                  <BreadcrumbLink asChild>
                    <Link to={p.href}>{p.name}</Link>
                  </BreadcrumbLink>
                ) : (
                  <BreadcrumbPage>{p.name}</BreadcrumbPage>
                )}
              </BreadcrumbItem>
              {idx < paths.length - 1 && <BreadcrumbSeparator />}
            </React.Fragment>
          ))}
        </BreadcrumbList>
      </Breadcrumb>
    </nav>
  );
};

export default AutoBreadcrumbs;
