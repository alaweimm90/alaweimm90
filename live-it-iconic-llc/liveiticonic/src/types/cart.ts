export interface CartItem {
  id: string;
  name: string;
  price: number;
  image: string;
  quantity: number;
  size?: string;
  color?: string;
  variant?: string;
}

export interface CartState {
  items: CartItem[];
  total: number;
  itemCount: number;
}

export interface CartContextType extends CartState {
  addItem: (item: Omit<CartItem, 'quantity'>) => void;
  removeItem: (id: string) => void;
  updateQuantity: (id: string, quantity: number) => void;
  clearCart: () => void;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}
