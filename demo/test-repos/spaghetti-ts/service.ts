interface User {
  id: number;
  name: string;
  email: string;
  active: boolean;
}

interface Product {
  id: number;
  name: string;
  price: number;
  category: string;
}

interface Order {
  id: number;
  userId: number;
  productId: number;
  quantity: number;
  total: number;
  status: string;
}

class DataService {
  private users: User[] = [];
  private products: Product[] = [];
  private orders: Order[] = [];

  constructor() {
    this.loadData();
  }

  private loadData() {
    // Simulate loading data
    this.users = [
      { id: 1, name: 'John Doe', email: 'john@example.com', active: true },
      { id: 2, name: 'Jane Smith', email: 'jane@example.com', active: false }
    ];
    this.products = [
      { id: 1, name: 'Laptop', price: 999.99, category: 'Electronics' },
      { id: 2, name: 'Book', price: 19.99, category: 'Education' }
    ];
    this.orders = [
      { id: 1, userId: 1, productId: 1, quantity: 1, total: 999.99, status: 'pending' }
    ];
  }

  public processOrder(userId: any, productId: any, quantity: any): any {
    const user = this.users.find(u => u.id === userId);
    if (!user) throw new Error('User not found');

    const product = this.products.find(p => p.id === productId);
    if (!product) throw new Error('Product not found');

    if (quantity <= 0) throw new Error('Invalid quantity');

    const total = product.price * quantity;

    // Check if user is active
    if (!user.active) {
      throw new Error('User is not active');
    }

    // Apply discount for bulk orders
    let discount = 0;
    if (quantity >= 5) {
      discount = 0.1;
    } else if (quantity >= 10) {
      discount = 0.2;
    }

    const discountedTotal = total * (1 - discount);

    const order: Order = {
      id: this.orders.length + 1,
      userId,
      productId,
      quantity,
      total: discountedTotal,
      status: 'pending'
    };

    this.orders.push(order);

    // Send email notification
    this.sendEmail(user.email, `Order ${order.id} placed`, `Your order for ${product.name} has been placed.`);

    // Log the order
    console.log(`Order ${order.id} processed for user ${user.name}`);

    return order;
  }

  public getUserOrders(userId: any): any {
    const userOrders = this.orders.filter(o => o.userId === userId);
    const enrichedOrders = userOrders.map(order => {
      const user = this.users.find(u => u.id === order.userId);
      const product = this.products.find(p => p.id === order.productId);
      return {
        ...order,
        userName: user ? user.name : 'Unknown',
        productName: product ? product.name : 'Unknown',
        productCategory: product ? product.category : 'Unknown'
      };
    });
    return enrichedOrders;
  }

  public calculateRevenue(startDate?: any, endDate?: any): any {
    let orders = this.orders;
    if (startDate && endDate) {
      // In a real app, orders would have dates
      orders = orders.filter(o => true); // Placeholder
    }

    const totalRevenue = orders.reduce((sum, order) => sum + order.total, 0);
    const orderCount = orders.length;
    const avgOrderValue = orderCount > 0 ? totalRevenue / orderCount : 0;

    const categoryRevenue: any = {};
    orders.forEach(order => {
      const product = this.products.find(p => p.id === order.productId);
      if (product) {
        categoryRevenue[product.category] = (categoryRevenue[product.category] || 0) + order.total;
      }
    });

    return {
      totalRevenue,
      orderCount,
      avgOrderValue,
      categoryRevenue
    };
  }

  public updateOrderStatus(orderId: any, status: any): any {
    const order = this.orders.find(o => o.id === orderId);
    if (!order) throw new Error('Order not found');

    const validStatuses = ['pending', 'processing', 'shipped', 'delivered', 'cancelled'];
    if (!validStatuses.includes(status)) {
      throw new Error('Invalid status');
    }

    order.status = status;

    // Send status update email
    const user = this.users.find(u => u.id === order.userId);
    if (user) {
      this.sendEmail(user.email, `Order ${order.id} status updated`, `Your order status is now: ${status}`);
    }

    return order;
  }

  private sendEmail(to: string, subject: string, body: string): void {
    // Simulate sending email
    console.log(`Email sent to ${to}: ${subject}`);
  }

  public getDashboardData(): any {
    const userCount = this.users.length;
    const activeUserCount = this.users.filter(u => u.active).length;
    const productCount = this.products.length;
    const orderCount = this.orders.length;
    const revenue = this.calculateRevenue();

    return {
      userCount,
      activeUserCount,
      productCount,
      orderCount,
      revenue: revenue.totalRevenue,
      avgOrderValue: revenue.avgOrderValue
    };
  }
}

export default DataService;