import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-{{COMPONENT_NAME}}',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './component.html',
  styleUrls: ['./component.scss'],
})
export class {{COMPONENT_NAME}}Component {
  @Input() title = '';
}
