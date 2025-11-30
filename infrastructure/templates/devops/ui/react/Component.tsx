import React from 'react';

interface {{COMPONENT_NAME}}Props {
  title?: string;
  children?: React.ReactNode;
}

export function {{COMPONENT_NAME}}({ title, children }: {{COMPONENT_NAME}}Props): React.ReactElement {
  return (
    <div className="{{COMPONENT_NAME}}">
      {title && <h2>{title}</h2>}
      {children}
    </div>
  );
}

export default {{COMPONENT_NAME}};
