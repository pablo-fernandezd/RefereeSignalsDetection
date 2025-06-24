import React from 'react';
import './NotificationSystem.css';

const NotificationSystem = ({ notifications }) => {
    if (!notifications || notifications.length === 0) return null;

    return (
        <div className="notification-container">
            {notifications.map((notification) => (
                <div 
                    key={notification.id} 
                    className={`notification notification-${notification.type}`}
                >
                    <div className="notification-content">
                        <span className="notification-icon">
                            {notification.type === 'success' && '✓'}
                            {notification.type === 'error' && '✗'}
                            {notification.type === 'warning' && '⚠'}
                            {notification.type === 'info' && 'ℹ'}
                        </span>
                        <span className="notification-message">{notification.message}</span>
                    </div>
                </div>
            ))}
        </div>
    );
};

export default NotificationSystem; 